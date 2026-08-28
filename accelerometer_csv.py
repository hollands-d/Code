"""Canonical CSV loading for converted-g and raw VMM accelerometer logs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = ("time_s", "X_g", "Y_g", "Z_g")
RAW_COUNT_COLUMNS = ("x_counts", "y_counts", "z_counts")


@dataclass
class DatasetMetadata:
    source_format: str
    reported_fs_g: tuple[float, ...] = ()
    reported_odr_hz: tuple[float, ...] = ()
    status_fault_count: int = 0
    timestamp_gap_count: int = 0
    original_columns: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)

    def description(self) -> str:
        return ("Raw VMM counts converted to g" if self.source_format == "raw_vmm_counts"
                else "Converted acceleration (g)")


def _column_lookup(columns: Iterable[object]) -> dict[str, object]:
    return {str(column).strip().lower(): column for column in columns}


def detect_csv_format(df: pd.DataFrame) -> str:
    low = _column_lookup(df.columns)
    if all(name in low for name in RAW_COUNT_COLUMNS):
        return "raw_vmm_counts"
    return "converted_g"


def _find_column(df: pd.DataFrame, aliases: Iterable[str]):
    columns = list(df.columns)
    low = _column_lookup(columns)
    for alias in aliases:
        if alias in low:
            return low[alias]
    for column in columns:
        text = str(column).lower()
        if any(alias in text for alias in aliases):
            return column
    return None


def _numeric(df: pd.DataFrame, column, label: str) -> np.ndarray:
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(values)):
        bad = int(np.sum(~np.isfinite(values)))
        raise ValueError(f"{label} contains {bad} missing or non-numeric value(s).")
    return values


def _positive_unique(values: np.ndarray, label: str) -> tuple[float, ...]:
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"Invalid {label}: values must be finite and greater than zero.")
    return tuple(float(value) for value in np.unique(values))


def _gap_count(timestamps_s: np.ndarray, expected_period_s: np.ndarray | None) -> int:
    if len(timestamps_s) < 2:
        return 0
    spacing = np.diff(timestamps_s)
    if np.any(spacing <= 0):
        return int(np.sum(spacing <= 0))
    if expected_period_s is not None:
        expected = expected_period_s[:-1]
    else:
        expected = np.full(len(spacing), float(np.median(spacing)))
    tolerance = np.maximum(expected * 0.5, 2e-6)
    return int(np.sum(np.abs(spacing - expected) > tolerance))


def _normalize_converted(df: pd.DataFrame, fallback_fs: float):
    time_col = _find_column(df, ("time_s", "time", "timestamp"))
    axis_cols = {
        "X_g": _find_column(df, ("x_g", "accel_x", "acc_x")),
        "Y_g": _find_column(df, ("y_g", "accel_y", "acc_y")),
        "Z_g": _find_column(df, ("z_g", "accel_z", "acc_z")),
    }
    if not all(axis_cols.values()):
        raise ValueError("Expected X/Y/Z acceleration columns such as X_g, Y_g and Z_g.")
    if not np.isfinite(fallback_fs) or fallback_fs <= 0:
        raise ValueError("Fallback sample rate must be greater than zero.")
    axes = {name: _numeric(df, column, name) for name, column in axis_cols.items()}
    if time_col is not None:
        time_s = _numeric(df, time_col, "time")
        time_s = time_s - time_s[0] if len(time_s) else time_s
        positive = np.diff(time_s)
        positive = positive[np.isfinite(positive) & (positive > 0)]
        sample_rate = 1.0 / float(np.median(positive)) if len(positive) else float(fallback_fs)
    else:
        sample_rate = float(fallback_fs)
        time_s = np.arange(len(df), dtype=float) / sample_rate
    canonical = pd.DataFrame({"time_s": time_s, **axes})
    metadata = DatasetMetadata(
        source_format="converted_g",
        original_columns=tuple(str(column) for column in df.columns),
    )
    return canonical, sample_rate, metadata


def _normalize_raw_vmm(df: pd.DataFrame, fallback_fs: float):
    low = _column_lookup(df.columns)
    counts = {name: _numeric(df, low[name], name) for name in RAW_COUNT_COLUMNS}
    if "fs_g" not in low:
        raise ValueError("Raw VMM count CSV requires fs_g for safe count-to-g conversion.")
    fs_g = _numeric(df, low["fs_g"], "fs_g")
    fs_values = _positive_unique(fs_g, "fs_g")
    invalid_ranges = [value for value in fs_values if value not in (2.0, 4.0, 8.0, 16.0)]
    if invalid_ranges:
        raise ValueError(f"Invalid LIS2DUX12 fs_g value(s): {invalid_ranges}.")

    odr = _numeric(df, low["odr_hz"], "odr_hz") if "odr_hz" in low else None
    odr_values = _positive_unique(odr, "odr_hz") if odr is not None else ()
    periods_us = (_numeric(df, low["sample_period_us"], "sample_period_us")
                  if "sample_period_us" in low else None)
    if periods_us is not None:
        _positive_unique(periods_us, "sample_period_us")

    if "sample_timestamp_us" in low:
        timestamps_us = _numeric(df, low["sample_timestamp_us"], "sample_timestamp_us")
        time_s = (timestamps_us - timestamps_us[0]) / 1_000_000.0
    elif periods_us is not None:
        time_s = np.zeros(len(df), dtype=float)
        if len(df) > 1:
            time_s[1:] = np.cumsum(periods_us[:-1]) / 1_000_000.0
    else:
        sample_rate = float(np.median(odr)) if odr is not None else float(fallback_fs)
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("Cannot construct raw VMM time: no valid timestamp, period or sample rate.")
        time_s = np.arange(len(df), dtype=float) / sample_rate

    expected_period_s = periods_us / 1_000_000.0 if periods_us is not None else None
    gap_count = _gap_count(time_s, expected_period_s)
    status_fault_count = 0
    if "status" in low:
        status = _numeric(df, low["status"], "status").astype(np.int64)
        nonzero = status != 0
        if "block_seq" in low:
            block_seq = _numeric(df, low["block_seq"], "block_seq").astype(np.int64)
            status_fault_count = len(np.unique(block_seq[nonzero]))
        else:
            status_fault_count = int(np.sum(nonzero))

    warnings = []
    if len(fs_values) > 1:
        warnings.append(f"full-scale changed {len(fs_values)-1} time(s); row-by-row scaling was applied")
    if len(odr_values) > 1:
        warnings.append(f"ODR changed {len(odr_values)-1} time(s); sample timestamps were retained")
    if status_fault_count:
        warnings.append(f"{status_fault_count} non-zero status block(s)")
    if gap_count:
        warnings.append(f"{gap_count} timestamp gap(s)")

    scale = fs_g / 32768.0
    canonical = pd.DataFrame({
        "time_s": time_s,
        "X_g": counts["x_counts"] * scale,
        "Y_g": counts["y_counts"] * scale,
        "Z_g": counts["z_counts"] * scale,
    })
    if odr is not None:
        sample_rate = float(np.median(odr))
    elif periods_us is not None:
        sample_rate = 1_000_000.0 / float(np.median(periods_us))
    else:
        sample_rate = float(fallback_fs)
    metadata = DatasetMetadata(
        source_format="raw_vmm_counts",
        reported_fs_g=fs_values,
        reported_odr_hz=odr_values,
        status_fault_count=status_fault_count,
        timestamp_gap_count=gap_count,
        original_columns=tuple(str(column) for column in df.columns),
        warnings=warnings,
    )
    return canonical, sample_rate, metadata


def normalize_accelerometer_dataframe(df: pd.DataFrame, fallback_fs: float = 200.0):
    """Return canonical time_s/X_g/Y_g/Z_g data, sample rate and metadata."""
    if df.empty:
        raise ValueError("CSV contains no accelerometer samples.")
    if detect_csv_format(df) == "raw_vmm_counts":
        return _normalize_raw_vmm(df, fallback_fs)
    return _normalize_converted(df, fallback_fs)
