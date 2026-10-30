"""Step 4a — Parse: turn the ESP32's text lines into CSI amplitudes.

The `csi_recv_router` firmware prints one line per received packet:

    CSI_DATA,<id>,<mac>,<rssi>,...,<local_timestamp>,...,<len>,<first_word>,"[b0,b1,...]"

The quoted list is the raw CSI buffer: `len` signed 8-bit integers (int8).
With only LLTF enabled (the firmware's setting) `len` = 128 = 64 subcarriers x 2 bytes.

Facts taken from the ESP-IDF "Wi-Fi Channel State Information" docs, and checked
against the real sample printed in the example's README (tests/esp32_readme_sample.csv):

* Each subcarrier is 2 bytes: IMAGINARY part first, then REAL part.
* Order of the 64 LLTF entries depends on `secondary_channel`:
    - 0 (none):          subcarriers  0..31, then -32..-1   (DC first, needs a swap)
    - 1 (above)/2 (below): already ascending  -32..31        (DC in the middle)
* If `first_word` (first_word_invalid) is 1, the first 4 bytes = first 2 entries are garbage
  (ESP32 hardware limitation).
* 802.11 legacy (OFDM) training symbols only use subcarriers -26..-1 and 1..26 -> 52 kept.
  -32..-27 and 27..31 are guard bands, 0 is DC: always (close to) zero, so we drop them.

Each packet becomes a `CsiPacket`; packets are grouped per transmitter MAC address into
`LinkData` (one "link" = one TX->RX pair), so the pipeline works for N links from day one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Default column list for the classic ESP32 firmware (used if the header line was missed).
DEFAULT_HEADER = (
    "type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,"
    "fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,"
    "sig_len,rx_format,len,first_word,data"
).split(",")

N_LLTF = 64
# Subcarrier index k (-32..31) of each of the 64 LLTF entries, per secondary-channel setting.
_K_NONE = np.r_[np.arange(0, 32), np.arange(-32, 0)]
_K_ASCENDING = np.arange(-32, 32)
# The 52 subcarriers that actually carry the training signal.
DATA_SUBCARRIERS = np.r_[np.arange(-26, 0), np.arange(1, 27)]

TIMESTAMP_WRAP_US = 2**32  # local_timestamp is a 32-bit microsecond counter (~71.6 min)


@dataclass
class CsiPacket:
    mac: str
    seq: int
    rssi: int
    timestamp_us: int          # ESP32 hardware RX timestamp (microseconds)
    secondary_channel: int
    first_word_invalid: bool
    csi: np.ndarray            # complex, 64 entries in raw buffer order
    host_time: float | None = None  # laptop time when the line arrived (if recorded)


@dataclass
class LinkData:
    """All packets of one TX->RX link, as arrays (T = number of packets, K = 52)."""
    mac: str
    t: np.ndarray              # seconds since first packet (from ESP32 timestamps)
    amp: np.ndarray            # T x K amplitude, columns = DATA_SUBCARRIERS
    rssi: np.ndarray
    seq: np.ndarray
    host_t: np.ndarray         # laptop time per packet (NaN if the file has none)


@dataclass
class Recording:
    links: dict[str, LinkData]
    meta: dict = field(default_factory=dict)
    path: Path | None = None

    @property
    def main_link(self) -> LinkData:
        """The link with the most packets (in router mode there is only one)."""
        return max(self.links.values(), key=lambda l: len(l.t))


def parse_line(line: str, header: list[str] = DEFAULT_HEADER) -> CsiPacket | None:
    """Parse one serial line. Returns None for anything that isn't a valid CSI line
    (boot logs, the header, truncated lines...)."""
    line = line.strip()
    host_time = None
    if not line.startswith("CSI_DATA"):
        # Our recorder prefixes "<host_time>," — strip it if present.
        head, _, rest = line.partition(",")
        if not rest.startswith("CSI_DATA"):
            return None
        try:
            host_time = float(head)
        except ValueError:
            return None
        line = rest
    meta_part, sep, data_part = line.partition(',"[')
    if not sep or not data_part.endswith(']"'):
        return None  # truncated line (common when serial bytes are lost)
    fields = dict(zip(header, meta_part.split(",")))
    try:
        raw = np.array(data_part[:-2].split(","), dtype=np.int16)
        n = int(fields["len"])
    except (ValueError, KeyError):
        return None
    if n != 2 * N_LLTF or raw.size != n:
        return None  # only LLTF-only packets (128 bytes) are expected from this firmware
    imag, real = raw[0::2].astype(float), raw[1::2].astype(float)
    return CsiPacket(
        mac=fields["mac"],
        seq=int(fields.get("id", fields.get("seq", -1))),
        rssi=int(fields["rssi"]),
        timestamp_us=int(fields["local_timestamp"]),
        secondary_channel=int(fields["secondary_channel"]),
        first_word_invalid=fields.get("first_word", "0") == "1",
        csi=real + 1j * imag,
        host_time=host_time,
    )


def select_subcarriers(pkt: CsiPacket) -> np.ndarray:
    """Reorder the 64 raw entries to subcarrier index and keep the 52 data subcarriers.
    Returns complex values (NaN where the hardware marked data invalid)."""
    k_of_entry = _K_NONE if pkt.secondary_channel == 0 else _K_ASCENDING
    csi = pkt.csi.copy()
    if pkt.first_word_invalid:
        csi[:2] = np.nan
    by_k = np.empty(N_LLTF, complex)
    by_k[k_of_entry + 32] = csi
    return by_k[DATA_SUBCARRIERS + 32]


def amplitude(csi: np.ndarray) -> np.ndarray:
    """|H(k)| = sqrt(re^2 + im^2). NaN entries (invalid first word) are filled by linear
    interpolation from neighbouring subcarriers — the channel is smooth across frequency."""
    amp = np.abs(csi)
    bad = np.isnan(amp)
    if bad.any() and not bad.all():
        idx = np.arange(amp.size)
        amp[bad] = np.interp(idx[bad], idx[~bad], amp[~bad])
    return amp


def unwrap_timestamps(ts_us: np.ndarray) -> np.ndarray:
    """The 32-bit microsecond counter wraps every ~71.6 minutes; make it monotonic."""
    ts = ts_us.astype(np.int64)
    jumps = np.diff(ts) < -(TIMESTAMP_WRAP_US // 2)
    return ts + TIMESTAMP_WRAP_US * np.r_[0, np.cumsum(jumps)]


def packets_to_links(packets: list[CsiPacket]) -> dict[str, LinkData]:
    by_mac: dict[str, list[CsiPacket]] = {}
    for p in packets:
        by_mac.setdefault(p.mac, []).append(p)
    links = {}
    for mac, pkts in by_mac.items():
        ts = unwrap_timestamps(np.array([p.timestamp_us for p in pkts]))
        links[mac] = LinkData(
            mac=mac,
            t=(ts - ts[0]) / 1e6,
            amp=np.array([amplitude(select_subcarriers(p)) for p in pkts]),
            rssi=np.array([p.rssi for p in pkts]),
            seq=np.array([p.seq for p in pkts]),
            host_t=np.array([np.nan if p.host_time is None else p.host_time for p in pkts]),
        )
    return links


def load_recording(path: str | Path) -> Recording:
    """Read a raw capture file (+ its .json sidecar with label/markers, if any)."""
    path = Path(path)
    header = DEFAULT_HEADER
    packets = []
    with open(path, errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("type,"):
                header = stripped.split(",")
                continue
            pkt = parse_line(stripped, header)
            if pkt is not None:
                packets.append(pkt)
    if not packets:
        raise ValueError(f"no CSI packets found in {path}")
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return Recording(links=packets_to_links(packets), meta=meta, path=path)
