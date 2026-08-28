"""Minimal UART2 COBS+CRC32 client for hw_test_vmm live accel streaming.

Engineer handoff: see tools/hw-diagnostics/ACCELEROMETER_STREAMING.md
(git tag: accelerometer-streaming).

Matches vibration-test-ctr / docs/vibration_motion_monitor protocol v1.
Default path: ST-LINK Virtual COM @ 460800 (USART2 DebugRnD).

Embed: copy this file only; depends on pyserial. No ProtoComms / GUI required.
"""

from __future__ import annotations

import struct
import time
import zlib
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple


SYNC = 0xA5
PROTO_VER = 1
PKT_HOST_CMD = 0x01
PKT_HOST_ACK = 0x02
PKT_SAMPLE = 0x10
PKT_CONFIG = 0x12
CMD_START = 0x01
CMD_STOP = 0x02
CMD_CONFIGURE = 0x03
CMD_STATUS = 0x05
DEFAULT_BAUD = 460800

CFG_FLAG_HIGH_PERFORMANCE = 0x01
CFG_FLAG_BW_SHIFT = 1
CFG_FLAG_BW_MASK = 0x06
CFG_FLAG_INT1_FIFO_TH = 0x08
CFG_FLAG_EXTENDED_VALID = 0x80
CONFIG_STRUCT = struct.Struct("<HBBIHHHHffffff")


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def _cobs_encode(data: bytes) -> bytes:
    if not data:
        return b"\x01"
    out = bytearray([0])
    code_i = 0
    code = 1
    for b in data:
        if b == 0:
            out[code_i] = code
            code_i = len(out)
            out.append(0)
            code = 1
        else:
            out.append(b)
            code += 1
            if code == 0xFF:
                out[code_i] = code
                code_i = len(out)
                out.append(0)
                code = 1
    out[code_i] = code
    return bytes(out)


def _cobs_decode(data: bytes) -> bytes:
    if not data:
        return b""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        code = data[i]
        if code == 0:
            raise ValueError("invalid COBS")
        i += 1
        end = i + code - 1
        if end > n:
            raise ValueError("COBS overrun")
        out.extend(data[i:end])
        i = end
        if code < 0xFF and i < n:
            out.append(0)
    return bytes(out)


def _pack_frame(packet: bytes) -> bytes:
    return _cobs_encode(packet + struct.pack("<I", _crc32(packet))) + b"\x00"


def _host_cmd(cmd: int, args: bytes = b"", seq: int = 0) -> bytes:
    payload = struct.pack("<BBH", cmd & 0xFF, len(args) & 0xFF, 0) + args
    hdr = struct.pack(
        "<BBBBHH",
        SYNC,
        PROTO_VER,
        PKT_HOST_CMD,
        0,
        seq & 0xFFFF,
        len(payload) & 0xFFFF,
    )
    return _pack_frame(hdr + payload)


@dataclass
class SampleBlock:
    seq: int
    timestamp_us: int
    sample_period_us: int
    odr_hz: int
    fs_g: int
    status: int
    xyz: List[Tuple[int, int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class AccelerometerConfig:
    """Runtime LIS2DUX12 acquisition settings carried by VMM CONFIG."""

    odr_hz: int = 200
    fs_g: int = 2
    high_performance: bool = True
    bandwidth_divisor: int = 2
    fifo_watermark: int = 32
    int1_fifo_threshold: bool = True
    stream_timeout_ms: int = 60_000

    def validate(self) -> None:
        if self.odr_hz not in (25, 50, 100, 200, 400, 800):
            raise ValueError("ODR must be 25, 50, 100, 200, 400 or 800 Hz")
        if self.fs_g not in (2, 4, 8, 16):
            raise ValueError("full scale must be 2, 4, 8 or 16 g")
        if self.bandwidth_divisor not in (2, 4, 8, 16):
            raise ValueError("bandwidth divisor must be 2, 4, 8 or 16")
        if not self.high_performance and self.odr_hz == 25 and self.bandwidth_divisor == 2:
            raise ValueError("25 Hz low-power mode does not support ODR/2 bandwidth")
        if not 1 <= self.fifo_watermark <= 32:
            raise ValueError("FIFO watermark must be between 1 and 32 samples")
        if not 1 <= self.stream_timeout_ms <= 0xFFFFFFFF:
            raise ValueError("stream timeout must be between 1 and 4294967295 ms")

    @property
    def flags(self) -> int:
        self.validate()
        bw_code = {2: 0, 4: 1, 8: 2, 16: 3}[self.bandwidth_divisor]
        flags = CFG_FLAG_EXTENDED_VALID | (bw_code << CFG_FLAG_BW_SHIFT)
        if self.high_performance:
            flags |= CFG_FLAG_HIGH_PERFORMANCE
        if self.int1_fifo_threshold:
            flags |= CFG_FLAG_INT1_FIFO_TH
        return flags

    def payload_bytes(self) -> bytes:
        return CONFIG_STRUCT.pack(
            self.odr_hz,
            self.fs_g,
            self.flags,
            self.stream_timeout_ms,
            self.fifo_watermark,
            0,
            1,
            0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
        )

    @classmethod
    def from_payload(cls, payload: bytes) -> "AccelerometerConfig":
        if len(payload) < CONFIG_STRUCT.size:
            raise ValueError("CONFIG payload truncated")
        values = CONFIG_STRUCT.unpack_from(payload)
        flags = values[2]
        if not flags & CFG_FLAG_EXTENDED_VALID:
            raise ValueError("CONFIG read-back lacks extended-valid flag")
        bw_code = (flags & CFG_FLAG_BW_MASK) >> CFG_FLAG_BW_SHIFT
        config = cls(
            odr_hz=values[0],
            fs_g=values[1],
            high_performance=bool(flags & CFG_FLAG_HIGH_PERFORMANCE),
            bandwidth_divisor=(2, 4, 8, 16)[bw_code],
            fifo_watermark=values[4],
            int1_fifo_threshold=bool(flags & CFG_FLAG_INT1_FIFO_TH),
            stream_timeout_ms=values[3],
        )
        config.validate()
        return config


@dataclass
class StreamStats:
    bytes_rx: int = 0
    frames_ok: int = 0
    frames_bad: int = 0
    acks: int = 0
    sample_blocks: int = 0
    samples: int = 0
    configs: int = 0
    last_config: Optional[AccelerometerConfig] = None
    last_xyz: Optional[Tuple[float, float, float]] = None  # g
    last_error: str = ""


class VmmStreamClient:
    """Open ST-LINK VCP, start/stop streaming, pull sample blocks."""

    def __init__(self, port: str, baud: int = DEFAULT_BAUD) -> None:
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pyserial required: pip install pyserial") from exc
        self._ser = serial.Serial(port, int(baud), timeout=0.05)
        self._seq = 0
        self._buf = bytearray()
        self.stats = StreamStats()
        self.port = port
        self.baud = int(baud)

    def close(self) -> None:
        try:
            self.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._ser.close()
        except Exception:  # noqa: BLE001
            pass

    def _next_seq(self) -> int:
        s = self._seq & 0xFFFF
        self._seq = (self._seq + 1) & 0xFFFF
        return s

    def start(self, timeout_ms: int = 120_000) -> None:
        """Start a new stream session; initial START clears stale VCP input bytes."""
        self._ser.reset_input_buffer()
        self._ser.write(_host_cmd(CMD_START, struct.pack("<I", int(timeout_ms)), self._next_seq()))
        self._ser.flush()

    def renew(self, timeout_ms: int = 120_000) -> None:
        """Renew the finite VMM START lease without clearing already-received serial data."""
        self._ser.write(_host_cmd(CMD_START, struct.pack("<I", int(timeout_ms)), self._next_seq()))
        self._ser.flush()

    def stop(self) -> None:
        self._ser.write(_host_cmd(CMD_STOP, b"", self._next_seq()))
        self._ser.flush()

    def configure(self, config: AccelerometerConfig) -> None:
        """Request runtime configuration; CONFIG read-back confirms success."""
        config.validate()
        self._ser.write(_host_cmd(CMD_CONFIGURE, config.payload_bytes(), self._next_seq()))
        self._ser.flush()

    def status(self) -> None:
        self._ser.write(_host_cmd(CMD_STATUS, b"", self._next_seq()))
        self._ser.flush()

    def _decode_packet(self, packet: bytes) -> Optional[object]:
        if len(packet) < 8:
            return None
        sync, ver, ptype, _r, _seq, plen = struct.unpack_from("<BBBBHH", packet)
        if sync != SYNC or ver != PROTO_VER:
            return None
        if 8 + plen > len(packet):
            return None
        payload = packet[8 : 8 + plen]
        if ptype == PKT_HOST_ACK and len(payload) >= 2:
            self.stats.acks += 1
            detail = struct.unpack_from("<H", payload, 2)[0] if len(payload) >= 4 else 0
            return ("ack", payload[0], payload[1], detail)
        if ptype == PKT_CONFIG:
            config = AccelerometerConfig.from_payload(payload)
            self.stats.configs += 1
            self.stats.last_config = config
            return config
        if ptype == PKT_SAMPLE and len(payload) >= 24:
            seq, ts, period, odr, fs, status, n, _res = struct.unpack_from(
                "<IQIHBBHH", payload
            )
            need = 24 + n * 6
            if len(payload) < need:
                return None
            xyz: List[Tuple[int, int, int]] = []
            off = 24
            for _ in range(n):
                x, y, z = struct.unpack_from("<hhh", payload, off)
                xyz.append((x, y, z))
                off += 6
            blk = SampleBlock(seq, ts, period, odr, fs, status, xyz)
            self.stats.sample_blocks += 1
            self.stats.samples += n
            if xyz:
                # Approx ±fs_g full-scale as int16
                scale = float(fs if fs else 2) / 32768.0
                x, y, z = xyz[-1]
                self.stats.last_xyz = (x * scale, y * scale, z * scale)
            return blk
        return None

    def poll(self) -> List[object]:
        chunk = self._ser.read(4096)
        if chunk:
            self.stats.bytes_rx += len(chunk)
            self._buf.extend(chunk)
        out: List[object] = []
        while True:
            try:
                i = self._buf.index(0)
            except ValueError:
                break
            frame = bytes(self._buf[:i])
            del self._buf[: i + 1]
            if not frame:
                continue
            try:
                decoded = _cobs_decode(frame)
                if len(decoded) < 4:
                    raise ValueError("short")
                packet, crc_b = decoded[:-4], decoded[-4:]
                (got,) = struct.unpack("<I", crc_b)
                if got != _crc32(packet):
                    raise ValueError("crc")
                self.stats.frames_ok += 1
                pkt = self._decode_packet(packet)
                if pkt is not None:
                    out.append(pkt)
            except Exception as exc:  # noqa: BLE001
                self.stats.frames_bad += 1
                self.stats.last_error = str(exc)
        return out

    def drain_for(self, duration_s: float) -> Iterator[object]:
        t_end = time.monotonic() + duration_s
        while time.monotonic() < t_end:
            for pkt in self.poll():
                yield pkt
            time.sleep(0.002)
