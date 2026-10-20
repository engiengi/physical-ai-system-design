#!/usr/bin/env bash
# Isaac Sim 5.1 driver for a fresh AWS L40S VM using NVIDIA's .run installer.
set -euo pipefail
COSMOS_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COSMOS_DRIVER_VERSION=580.65.06
[[ "$(uname -m)" == x86_64 && "$(cat /sys/devices/virtual/dmi/id/sys_vendor)" == 'Amazon EC2' ]] || {
    echo 'This script supports AWS x86 VMs only. Do not run it on Spark or Thor.' >&2; exit 1;
}
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 22.04 ]] || {
    echo 'Use the Ubuntu 22.04 AWS VM configuration from the guide.' >&2; exit 1;
}
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader)" == 'NVIDIA L40S' ]] || {
    echo 'Expected exactly one NVIDIA L40S.' >&2; exit 1;
}
if [[ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)" == "$COSMOS_DRIVER_VERSION" ]]; then
    nvidia-smi; exit 0
fi
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || {
    echo 'Stop all simulation and inference processes before changing the driver.' >&2; exit 1;
}
command -v nvidia-uninstall >/dev/null || {
    echo 'This VM uses a different driver installation method; do not mix package managers.' >&2; exit 1;
}
COSMOS_DRIVER_CACHE="$COSMOS_PROJECT/.cache/drivers"
mkdir -p "$COSMOS_DRIVER_CACHE"
COSMOS_DRIVER_FILE="$COSMOS_DRIVER_CACHE/NVIDIA-Linux-x86_64-$COSMOS_DRIVER_VERSION.run"
curl -fL --retry 3 "https://us.download.nvidia.com/XFree86/Linux-x86_64/$COSMOS_DRIVER_VERSION/NVIDIA-Linux-x86_64-$COSMOS_DRIVER_VERSION.run" -o "$COSMOS_DRIVER_FILE"
sh "$COSMOS_DRIVER_FILE" --check
sudo apt-get install -y build-essential dkms "linux-headers-$(uname -r)"
for COSMOS_SERVICE in nvidia-dcgm.service nvidia-cdi-refresh.path nvidia-cdi-refresh.service; do
    if systemctl cat "$COSMOS_SERVICE" >/dev/null 2>&1; then sudo systemctl stop "$COSMOS_SERVICE"; fi
done
sudo nvidia-smi --persistence-mode=0
sudo pkill -TERM -x nvidia-persiste || true
for COSMOS_MODULE in efa_nv_peermem gdrdrv nvidia_fs nvidia_uvm nvidia_drm nvidia_modeset nvidia; do
    if grep -q "^$COSMOS_MODULE " /proc/modules; then sudo rmmod "$COSMOS_MODULE"; fi
done
sudo sh "$COSMOS_DRIVER_FILE" --silent --dkms --kernel-module-type=open
sudo ldconfig
sudo modprobe nvidia
nvidia-smi
