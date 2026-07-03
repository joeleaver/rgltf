#!/usr/bin/env bash
# Build a PORTABLE rgltf-x86_64.AppImage by compiling the release binary inside an
# Ubuntu 20.04 (glibc 2.31) Docker container, then packaging it with build-appimage.sh.
#
# Why: an AppImage inherits the glibc of its build host. Built on a current distro the
# binary won't start on older ones ("GLIBC_2.xx not found"). Compiling on 20.04 drops the
# floor to glibc 2.31, so the result runs on Ubuntu 20.04+, Debian 11+, Fedora 32+, etc.
#
# Requires: Docker, and appimagetool (APPIMAGETOOL env var or on PATH — build-appimage.sh
# consumes it). The container does a full from-scratch dependency compile; the first run
# is slow. GPU drivers and the Vulkan loader are host-provided and are never bundled.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE="${BUILD_IMAGE:-ubuntu:20.04}"
CNAME="rgltf-portable-build"

# Steps run inside the container. Source is mounted read-only at /src; build artifacts go
# to the ephemeral /build (kept off the read-only mount). The GTK stack and libxdo are
# needed only at LINK time by muda (the menu library); --as-needed + --gc-sections drop
# them from the finished binary, which links only libc/libm/libstdc++/fontconfig (ldd).
read -r -d '' INNER <<'EOF' || true
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    curl git build-essential pkg-config ca-certificates \
    python3 python3-pip \
    libfontconfig1-dev libxkbcommon-dev libgtk-3-dev libxdo-dev
pip3 install --quiet --upgrade cmake   # 20.04's cmake (3.16) is too old for the KTX build
hash -r
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
export PATH="$HOME/.cargo/bin:$PATH"
cd /src
export CARGO_TARGET_DIR=/build/target
cargo build -p rgltf-app --release --locked
cp /build/target/release/rgltf /build/rgltf-portable
objdump -T /build/rgltf-portable | grep -oE 'GLIBC_[0-9.]+' | sort -V | uniq -c
EOF

docker rm -f "$CNAME" >/dev/null 2>&1 || true
docker run --name "$CNAME" -v "$REPO:/src:ro" "$IMAGE" bash -c "$INNER"

# Extract the binary (docker cp restores the invoking user's ownership) into the spot
# build-appimage.sh reads, then package.
install -d "$REPO/target/release"
docker cp "$CNAME:/build/rgltf-portable" "$REPO/target/release/rgltf"
docker rm -f "$CNAME" >/dev/null 2>&1 || true
chmod 755 "$REPO/target/release/rgltf"

"$REPO/assets/linux/appimage/build-appimage.sh"
echo "portable AppImage built from $IMAGE"
