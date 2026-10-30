#!/usr/bin/env bash
# One-time install of ESP-IDF (Espressif IoT Development Framework) for the classic ESP32.
# Needs ~3 GB of free disk space and an internet connection. Takes 10-20 minutes.
#
#   bash firmware/setup_esp_idf.sh
#
# Afterwards, in every new terminal where you want to build/flash:
#   . ~/esp/esp-idf/export.sh
set -euo pipefail

IDF_VERSION="v5.5.6"
IDF_DIR="$HOME/esp/esp-idf"

free_gb=$(df -g "$HOME" | awk 'NR==2 {print $4}')
if [ "$free_gb" -lt 3 ]; then
  echo "Only ${free_gb} GB free on the disk — ESP-IDF needs ~3 GB. Free some space first." >&2
  exit 1
fi

# Build tools (cmake is already installed; ninja and dfu-util are not)
brew install cmake ninja dfu-util

mkdir -p "$HOME/esp"
if [ ! -d "$IDF_DIR" ]; then
  # shallow clone keeps the download small
  git clone -b "$IDF_VERSION" --depth 1 --recursive --shallow-submodules \
      https://github.com/espressif/esp-idf.git "$IDF_DIR"
fi

# Compiler toolchain + Python environment, only for the ESP32 target
"$IDF_DIR/install.sh" esp32

echo
echo "Done. In a new terminal run:  . $IDF_DIR/export.sh"
echo "then follow firmware/README.md"
