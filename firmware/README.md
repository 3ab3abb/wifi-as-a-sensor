# Firmware — ESP32 side (Phase 1: router / ping mode)

`csi_recv_router/` is Espressif's example, copied unchanged from
[espressif/esp-csi](https://github.com/espressif/esp-csi) (`examples/get-started/csi_recv_router`,
Apache-2.0, see `LICENSE-espressif`). It does exactly what our design rule asks: the ESP32 only
**streams raw CSI**, all processing happens in Python. We do *not* use `esp-radar`.

What it does:
1. Connects to your router as a normal Wi-Fi station (2.4 GHz network only — the ESP32 has no 5 GHz).
2. Pings the router 100 times per second (`CONFIG_SEND_FREQUENCY 100` in `main/app_main.c`).
3. For every reply from the router, the Wi-Fi chip measures the channel (CSI) and the callback
   prints one `CSI_DATA,...` line over USB serial at 921600 baud.
4. Only the LLTF (legacy long training field) is enabled → 64 subcarriers × 2 bytes = 128 values
   per line. On the classic ESP32 no gain compensation is applied (that code is only for S3/C3/C5/C6),
   so AGC jumps are still in the data — our Python normalizes them.

## One-time setup (laptop)

```bash
bash firmware/setup_esp_idf.sh      # installs ESP-IDF v5.5.6 into ~/esp/esp-idf (~3 GB)
```

## Build (no board needed)

```bash
. ~/esp/esp-idf/export.sh           # in every new terminal
cd firmware/csi_recv_router
idf.py set-target esp32
idf.py menuconfig                   # "Example Connection Configuration" → WiFi SSID + password
idf.py build
```

The Wi-Fi name and password end up in `sdkconfig`, which is git-ignored so your password
is never committed.

## Flash + check (board needed)

```bash
python -m serial.tools.list_ports   # find the port, e.g. /dev/cu.usbserial-0001
idf.py -p /dev/cu.usbserial-XXXX flash monitor      # Ctrl+] quits the monitor
```

Expected: boot logs, `got ip:...`, then `CSI_DATA,...` lines scrolling fast (~100 per second)
→ **checkpoint step 2**. Then close the monitor (only one program can use the port) and record
from Python — see `python/README.md`.

If no port appears when the board is plugged in: try another USB cable (many are charge-only),
and check the USB-to-serial chip on the board (CP2102 or CH340); macOS usually has a driver
built in, otherwise install the vendor's driver.
