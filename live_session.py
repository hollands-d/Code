"""Monotonic operator-timeout, VMM renewal and stream-stall state."""
from __future__ import annotations

from dataclasses import dataclass, field
import threading


@dataclass(frozen=True)
class OperatorTimeoutConfig:
    enabled: bool = True
    duration_s: float = 300.0
    action: str = "Ask"

    def validate(self) -> None:
        if self.enabled and self.duration_s <= 0:
            raise ValueError("Capture timeout duration must be greater than zero.")
        if self.action not in ("Ask", "Continue same duration", "Continue indefinitely", "Stop"):
            raise ValueError(f"Unsupported capture-timeout action: {self.action}")


@dataclass
class LiveSessionState:
    operator: OperatorTimeoutConfig
    vmm_start_timeout_ms: int = 300_000
    vmm_renew_fraction: float = 0.8
    max_recovery_attempts: int = 3
    started_monotonic: float = 0.0
    operator_deadline: float | None = None
    next_vmm_renewal_monotonic: float = 0.0
    last_sample_block_monotonic: float = 0.0
    next_recovery_monotonic: float = 0.0
    stream_renewals: int = 0
    timeout_extensions: int = 0
    recovery_attempts: int = 0
    operator_indefinite: bool = False
    timeout_prompt_pending: bool = False
    stalled: bool = False
    recovery_failed: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def start(self, now: float) -> None:
        self.operator.validate()
        if self.vmm_start_timeout_ms <= 0:
            raise ValueError("VMM START timeout must be finite and greater than zero.")
        if not 0.1 <= self.vmm_renew_fraction < 1.0:
            raise ValueError("VMM renewal fraction must be between 0.1 and 1.0.")
        with self._lock:
            self.started_monotonic = now
            self.last_sample_block_monotonic = now
            self.next_recovery_monotonic = now
            self.next_vmm_renewal_monotonic = now + self._renew_interval_s()
            self.operator_indefinite = (not self.operator.enabled
                                        or self.operator.action == "Continue indefinitely")
            self.operator_deadline = (None if self.operator_indefinite
                                      else now + self.operator.duration_s)
            self.stream_renewals = 0
            self.timeout_extensions = 0
            self.recovery_attempts = 0
            self.timeout_prompt_pending = False
            self.stalled = False
            self.recovery_failed = False

    def _renew_interval_s(self) -> float:
        return self.vmm_start_timeout_ms / 1000.0 * self.vmm_renew_fraction

    def record_sample(self, now: float) -> bool:
        with self._lock:
            recovered = self.stalled
            self.last_sample_block_monotonic = now
            self.stalled = False
            self.recovery_failed = False
            return recovered

    def due_actions(self, now: float, stall_after_s: float) -> list[str]:
        actions: list[str] = []
        with self._lock:
            if now >= self.next_vmm_renewal_monotonic:
                actions.append("renew_vmm")
                # Reserve the next slot now to avoid repeatedly queueing renewal.
                self.next_vmm_renewal_monotonic = now + self._renew_interval_s()

            if (self.operator_deadline is not None and now >= self.operator_deadline
                    and not self.timeout_prompt_pending):
                if self.operator.action == "Ask":
                    self.timeout_prompt_pending = True
                    actions.append("ask_operator")
                elif self.operator.action == "Continue same duration":
                    self.operator_deadline = now + self.operator.duration_s
                    self.timeout_extensions += 1
                    actions.append("operator_extended")
                elif self.operator.action == "Stop":
                    self.operator_deadline = None
                    actions.append("stop")

            age = now - self.last_sample_block_monotonic
            if age >= stall_after_s and now >= self.next_recovery_monotonic:
                self.stalled = True
                if self.recovery_attempts < self.max_recovery_attempts:
                    self.recovery_attempts += 1
                    self.next_recovery_monotonic = now + stall_after_s
                    actions.append("recover_stream")
                elif not self.recovery_failed:
                    self.recovery_failed = True
                    actions.append("recovery_failed")
        return actions

    def mark_vmm_renewal(self, now: float) -> None:
        with self._lock:
            self.stream_renewals += 1
            self.next_vmm_renewal_monotonic = now + self._renew_interval_s()

    def continue_same_duration(self, now: float) -> None:
        with self._lock:
            self.timeout_prompt_pending = False
            self.operator_deadline = now + self.operator.duration_s
            self.timeout_extensions += 1

    def continue_indefinitely(self) -> None:
        with self._lock:
            self.timeout_prompt_pending = False
            self.operator_indefinite = True
            self.operator_deadline = None

    def cancel_prompt_for_stop(self) -> None:
        with self._lock:
            self.timeout_prompt_pending = False
            self.operator_deadline = None

    def snapshot(self, now: float) -> dict[str, float | int | bool | None]:
        with self._lock:
            return {
                "elapsed_s": max(0.0, now - self.started_monotonic),
                "operator_remaining_s": (None if self.operator_deadline is None
                                          else max(0.0, self.operator_deadline - now)),
                "operator_indefinite": self.operator_indefinite,
                "vmm_remaining_s": max(0.0, self.next_vmm_renewal_monotonic - now),
                "last_block_age_s": max(0.0, now - self.last_sample_block_monotonic),
                "stream_renewals": self.stream_renewals,
                "timeout_extensions": self.timeout_extensions,
                "recovery_attempts": self.recovery_attempts,
                "stalled": self.stalled,
                "recovery_failed": self.recovery_failed,
            }
