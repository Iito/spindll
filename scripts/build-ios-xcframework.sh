#!/usr/bin/env bash
# Build SpindllFFI.xcframework (iPhone + Simulator) for the SpindllKit Swift
# package, from the spindll-ffi crate.
#
# The default is the keyboard build. llama.cpp leaves Metal out, because its
# setup alone costs 55 MB of a keyboard extension's ~76 MB even with every
# layer on the CPU. It also leaves weight repacking out, because repacking
# copies mmap'd weights into memory iOS counts against the process. Pass
# --metal for a containing-app build that keeps the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

metal=OFF
features=()
if [[ "${1:-}" == "--metal" ]]; then
  metal=ON
  features=(--features metal)
fi

# Match SpindllKit's iOS 17 floor, so nothing is built for a newer OS than
# the app links against. llama-cpp-sys-2 forwards CMAKE_* and GGML_* to CMake.
export IPHONEOS_DEPLOYMENT_TARGET=17.0
export CMAKE_OSX_DEPLOYMENT_TARGET=17.0
export GGML_METAL=$metal
export GGML_CPU_REPACK=OFF

for target in aarch64-apple-ios aarch64-apple-ios-sim; do
  # ${a[@]+"${a[@]}"}: macOS bash 3.2 treats an empty array as unbound under set -u.
  cargo build --release -p spindll-ffi --target "$target" ${features[@]+"${features[@]}"}
done

out=ios/SpindllKit/SpindllFFI.xcframework
rm -rf "$out"
xcodebuild -create-xcframework \
  -library target/aarch64-apple-ios/release/libspindll_ffi.a -headers ffi/include \
  -library target/aarch64-apple-ios-sim/release/libspindll_ffi.a -headers ffi/include \
  -output "$out"
echo "built $out (Metal $metal)"
