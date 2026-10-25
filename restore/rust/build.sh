#!/bin/bash
# Build restore/lenscore.dll (Windows, GNU toolchain: no Visual Studio needed) or liblenscore.so (Linux).
#   Windows: rustup with host x86_64-pc-windows-gnu; set RUSTUP_HOME / CARGO_HOME if installed elsewhere.
#   The GNU toolchain's own MinGW tools must be on PATH (rustlib/x86_64-pc-windows-gnu/bin/self-contained).
set -e
cd "$(dirname "$0")/lenscore"
if [ "$OS" = "Windows_NT" ]; then
  SC=$(ls -d "${RUSTUP_HOME:-$HOME/.rustup}"/toolchains/*windows-gnu/lib/rustlib/x86_64-pc-windows-gnu/bin/self-contained 2>/dev/null | head -1)
  [ -n "$SC" ] && export PATH="$SC:$PATH"
  cargo build --release --target x86_64-pc-windows-gnu
  cp target/x86_64-pc-windows-gnu/release/lenscore.dll ../../lenscore.dll
else
  cargo build --release
  cp target/release/liblenscore.so ../../liblenscore.so
fi
echo "built $(ls ../../*lenscore*)"
