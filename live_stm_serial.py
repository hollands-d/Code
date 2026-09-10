"""Direct STM32 accelerometer acquisition through the ST-Link Virtual COM port.

This follows the firmware_tool method in the supplied diagnostics branch:

Windows laptop
    -> ST-Link pod USB
    -> ST-Link Virtual COM Port (COMx)
    -> STM32 USART2 debug/R&D UART
    -> ProtoComms framing + protobuf
    -> accelerometer buffer messages

Important terminology:
The physical pod is connected through the STM32 debug connector and may also be
used for SWD/JTAG/SWO, but THIS diagnostics data path is the ST-Link Virtual COM
UART, not JTAG/SWD memory access.

The target firmware must be the HW_TEST/DEBUG_RND_UART variant because that
variant routes ProtoComms to the debug UART instead of the normal SOM link.
"""
from __future__ import annotations

import os
import sys
import struct
import threading
import time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

try:
    import serial
    from serial.tools import list_ports
except ImportError as exc:
    serial = None
    list_ports = None

# The generated bindings supplied in the diagnostics branch use the Python
# protobuf runtime and absolute imports between generated modules.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
_GEN = Path(__file__).resolve().parent / "stm_proto_generated"
if str(_GEN) not in sys.path:
    sys.path.insert(0, str(_GEN))

import WrapperMessage_pb2 as wrapper_pb2  # noqa: E402
import MessageVersion_pb2  # noqa: F401,E402
import MessageResponse_pb2  # noqa: F401,E402
import MessageAccelerometer_pb2  # noqa: F401,E402

DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT_MS = 3000


def crc32_mpeg2(data: bytes) -> int:
    """CRC-32/MPEG-2 matching the STM32 ProtoComms implementation."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
    return crc


def available_serial_ports() -> List[Tuple[str, str, str]]:
    """Return (device, description, hwid), with ST-Link-looking ports first."""
    if list_ports is None:
        return []
    rows = []
    for p in list_ports.comports():
        rows.append((p.device, p.description or "", p.hwid or ""))
    rows.sort(
        key=lambda r: (
            0 if ("STLINK" in (r[1] + " " + r[2]).upper() or
                  "ST-LINK" in (r[1] + " " + r[2]).upper()) else 1,
            r[0]
        )
    )
    return rows


class StLinkVcpSerial:
    """Length-prefixed ProtoComms transport over an ST-Link virtual COM port.

    On the wire each frame is ``uint32 length + payload + uint32 CRC`` in
    little-endian order. The CRC covers the encoded length and payload.
    """

    def __init__(self, port: str, baudrate: int = DEFAULT_BAUD) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self.ser = None

    @property
    def is_open(self) -> bool:
        return bool(self.ser and self.ser.is_open)

    def connect(self) -> None:
        if serial is None:
            raise RuntimeError("pyserial is required. Run: py -m pip install pyserial")
        port = self.port
        # pyserial handles normal COMx names on native Windows, but retain the
        # same explicit device form used by the colleague tool for robustness.
        if sys.platform == "win32":
            import re
            if re.fullmatch(r"COM\d+", port, re.IGNORECASE):
                port = rf"\\.\{port}"
        self.ser = serial.Serial(
            port=port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1.0,
            write_timeout=2.0,
        )
        # Clear any partial bytes left by a prior session.
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def disconnect(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None

    def send_payload(self, payload: bytes) -> None:
        """Frame and synchronously write one serialized protobuf payload."""
        if not self.is_open:
            raise RuntimeError("serial port is not connected")
        size_bytes = struct.pack("<I", len(payload))
        crc = crc32_mpeg2(size_bytes + payload)
        frame = size_bytes + payload + struct.pack("<I", crc)
        self.ser.write(frame)
        self.ser.flush()

    def _read_exact(self, n: int, deadline: float) -> bytes:
        """Read up to ``n`` bytes without extending the caller's deadline."""
        chunks = bytearray()
        while len(chunks) < n and time.monotonic() < deadline:
            remain = deadline - time.monotonic()
            self.ser.timeout = max(0.01, min(0.25, remain))
            part = self.ser.read(n - len(chunks))
            if part:
                chunks.extend(part)
        return bytes(chunks)

    def receive_payload(self, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> Optional[bytes]:
        """Read and validate one frame, returning ``None`` on an incomplete timeout."""
        if not self.is_open:
            raise RuntimeError("serial port is not connected")
        deadline = time.monotonic() + timeout_ms / 1000.0
        header = self._read_exact(4, deadline)
        if len(header) != 4:
            return None
        size = struct.unpack("<I", header)[0]
        if size == 0 or size > 4096:
            # This usually means the stream was not aligned to a ProtoComms
            # frame or the wrong COM port/firmware is in use.
            raise RuntimeError(f"implausible ProtoComms frame size {size}")
        payload = self._read_exact(size, deadline)
        if len(payload) != size:
            return None
        crc_raw = self._read_exact(4, deadline)
        if len(crc_raw) != 4:
            return None
        got = struct.unpack("<I", crc_raw)[0]
        calc = crc32_mpeg2(header + payload)
        if got != calc:
            raise RuntimeError(
                f"ProtoComms CRC mismatch: received 0x{got:08X}, calculated 0x{calc:08X}"
            )
        return payload


class DirectStm32Client:
    """Serialized protobuf request/response interface over ST-Link VCP."""

    def __init__(self) -> None:
        self.transport: Optional[StLinkVcpSerial] = None
        self._lock = threading.RLock()
        self._next_id = 0xD100

    @property
    def is_connected(self) -> bool:
        return bool(self.transport and self.transport.is_open)

    def _alloc_id(self) -> int:
        """Allocate a positive application message ID, wrapping at int32 max."""
        mid = self._next_id
        self._next_id = (self._next_id + 1) & 0x7FFFFFFF
        if self._next_id < 0xD100:
            self._next_id = 0xD100
        return mid

    def connect(self, port: str, baudrate: int = DEFAULT_BAUD) -> str:
        self.disconnect()
        tr = StLinkVcpSerial(port, baudrate)
        tr.connect()
        self.transport = tr
        try:
            version = self.get_version()
        except Exception:
            tr.disconnect()
            self.transport = None
            raise
        return version

    def disconnect(self) -> None:
        if self.transport is not None:
            try:
                self.transport.disconnect()
            finally:
                self.transport = None

    def _request(self, request_field: str, response_field: str,
                 timeout_ms: int = DEFAULT_TIMEOUT_MS):
        """Send one protobuf request and wait for its correlated typed response."""
        if not self.transport:
            raise RuntimeError("not connected")
        with self._lock:
            mid = self._alloc_id()
            req = wrapper_pb2.WrapperMessage()
            req.iProtocolVersion = 1
            req.iMsgID = mid
            getattr(req, request_field).bActive = True
            self.transport.send_payload(req.SerializeToString())

            deadline = time.monotonic() + timeout_ms / 1000.0
            while time.monotonic() < deadline:
                remaining = int(max(1, (deadline - time.monotonic()) * 1000))
                raw = self.transport.receive_payload(remaining)
                if raw is None:
                    continue
                msg = wrapper_pb2.WrapperMessage()
                msg.ParseFromString(raw)

                # Match the colleague firmware_tool behaviour: the STM emits an
                # immediate mResp ACK, then the application response.
                if msg.iMsgID != mid:
                    continue
                kind = msg.WhichOneof("msg")
                if kind == "mResp":
                    continue
                if kind == response_field:
                    return msg
                # Ignore other correlated asynchronous response types.

            raise TimeoutError(f"timeout waiting for {response_field}")

    def get_version(self) -> str:
        msg = self._request("mGetVersion", "mVersion")
        v = msg.mVersion
        return (
            f"{v.mFWVersion.iMajor}.{v.mFWVersion.iMinor} "
            f"{v.sVersionDetails}".strip()
        )

    def get_num_buffers(self) -> int:
        msg = self._request(
            "mGetAccelerometerNumBuffers",
            "mAccelerometerNumBuffers",
        )
        return int(msg.mAccelerometerNumBuffers.iBufferCount)

    def get_data_buffer(self) -> bytes:
        msg = self._request(
            "mGetAccelerometerDataMsg",
            "mAccelerometerDataMsg",
        )
        return bytes(msg.mAccelerometerDataMsg.mArrSamples)


def decode_accelerometer_buffer(payload: bytes, scale_mg_per_lsb: float) -> np.ndarray:
    """Decode packed little-endian int16 X/Y/Z triples into g."""
    if not payload:
        return np.empty((0, 3), dtype=float)
    if len(payload) % 6:
        raise ValueError(
            f"accelerometer buffer length {len(payload)} is not a multiple of 6 bytes"
        )
    raw = np.frombuffer(payload, dtype="<i2").reshape(-1, 3).astype(float)
    return raw * (float(scale_mg_per_lsb) / 1000.0)
