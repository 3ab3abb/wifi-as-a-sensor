"""Data sources — everything downstream only sees an iterator of text lines in the ESP32
CSV format, so swapping the source never touches the pipeline:

* SerialSource : a real ESP32 on USB (Phase 1 router mode, later Phase 2 ESP-NOW too)
* ReplaySource : a recorded file, played back in real time (test the live demo without hardware)

Each yields (host_time, line).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

from pipeline.parse import parse_line

DEFAULT_BAUD = 921600   # set in the firmware's sdkconfig.defaults


def find_esp32_port() -> str | None:
    """Best guess at the ESP32's serial port (USB-UART chips: CP210x, CH340, FTDI)."""
    from serial.tools import list_ports
    for p in list_ports.comports():
        name = (p.device + " " + (p.description or "")).lower()
        if any(s in name for s in ("usbserial", "slab_usbtouart", "wchusbserial", "cp210", "ch340", "usbmodem")):
            return p.device
    return None


class SerialSource:
    def __init__(self, port: str | None = None, baud: int = DEFAULT_BAUD):
        import serial
        port = port or find_esp32_port()
        if port is None:
            raise RuntimeError("No ESP32 serial port found — is it plugged in? (list: python -m serial.tools.list_ports)")
        self.port = port
        self.ser = serial.Serial(port, baud, timeout=1)

    def __iter__(self) -> Iterator[tuple[float, str]]:
        buf = b""
        while True:
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if not chunk:
                continue
            now = time.time()
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                yield now, raw.decode("ascii", errors="replace").strip()

    def close(self):
        self.ser.close()


class ReplaySource:
    """Plays a recorded file back at its original pace (speed=2 -> twice as fast)."""

    def __init__(self, path: str | Path, speed: float = 1.0, loop: bool = False):
        self.path, self.speed, self.loop = Path(path), speed, loop

    def __iter__(self) -> Iterator[tuple[float, str]]:
        while True:
            start_wall, start_ts = time.time(), None
            with open(self.path) as f:
                for line in f:
                    pkt = parse_line(line)
                    if pkt is None:
                        continue
                    if start_ts is None:
                        start_ts = pkt.timestamp_us
                    delay = ((pkt.timestamp_us - start_ts) % 2**32) / 1e6 / self.speed - (time.time() - start_wall)
                    if delay > 0:
                        time.sleep(delay)
                    yield time.time(), line.strip()
            if not self.loop:
                return

    def close(self):
        pass
