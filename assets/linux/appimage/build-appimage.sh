#!/usr/bin/env bash
# Build rgltf-x86_64.AppImage from a release build.
#
# Requires a release binary (target/release/rgltf) and appimagetool. Point at a
# specific appimagetool with the APPIMAGETOOL env var; otherwise `appimagetool`
# on PATH is used. Output: target/appimage/rgltf-x86_64.AppImage.
#
# Portability note: the AppImage relies on the HOST's Vulkan loader (libvulkan.so.1)
# and GPU driver ICDs — GPU drivers are host-specific and must not be bundled. Built
# on this distro, it also inherits this glibc's minimum; build in an older-glibc
# environment for wider compatibility.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

BIN="target/release/rgltf"
[ -x "$BIN" ] || { echo "error: $BIN missing — run: cargo build -p rgltf-app --release" >&2; exit 1; }

APPDIR="target/appimage/rgltf.AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR"

install -Dm755 "$BIN" "$APPDIR/usr/bin/rgltf"
install -Dm755 assets/linux/appimage/AppRun "$APPDIR/AppRun"
install -Dm644 assets/linux/rgltf.desktop "$APPDIR/rgltf.desktop"
install -Dm644 assets/linux/rgltf.desktop "$APPDIR/usr/share/applications/rgltf.desktop"
install -Dm644 assets/linux/rgltf.xml "$APPDIR/usr/share/mime/packages/rgltf.xml"
install -Dm644 assets/branding/rgltf.svg "$APPDIR/usr/share/icons/hicolor/scalable/apps/rgltf.svg"
for s in 16 24 32 48 64 128 256 512; do
    install -Dm644 "assets/branding/icons/${s}x${s}/rgltf.png" \
        "$APPDIR/usr/share/icons/hicolor/${s}x${s}/apps/rgltf.png"
done
# appimagetool derives .DirIcon from the top-level icon named by the .desktop's Icon=.
install -Dm644 assets/branding/icons/256x256/rgltf.png "$APPDIR/rgltf.png"

TOOL="${APPIMAGETOOL:-appimagetool}"
OUT="target/appimage/rgltf-x86_64.AppImage"
ARCH=x86_64 "$TOOL" "$APPDIR" "$OUT"
echo "built $OUT"
