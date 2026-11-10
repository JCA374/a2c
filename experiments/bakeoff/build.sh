#!/bin/bash
# Build every bake-off implementation whose toolchain is present into ./out/ (missing toolchains are skipped).
# Toolchains (user-local, no sudo):  Rust: curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
#                                    Go:   tar -C ~/.local -xzf go<ver>.linux-amd64.tar.gz  (https://go.dev/dl)
cd "$(dirname "$0")"; mkdir -p out
PATH="$PATH:$HOME/.cargo/bin:$HOME/.local/go/bin"
if command -v g++ >/dev/null; then
  g++ -O2 -fPIC -shared -ffp-contract=off -fno-fast-math -pthread kernel.cpp -o out/libkernel_cpp.so && echo "built C++"
  g++ -O3 -march=native -fPIC -shared -ffp-contract=off -fno-fast-math -pthread kernel.cpp -o out/libkernel_cpp_native.so && echo "built C++ -O3 -march=native"
else echo "skip C++ (no g++)"; fi
if command -v cargo >/dev/null; then
  (cd rust && cargo build --release -q) && cp rust/target/release/libkernel.so out/libkernel_rust.so && echo "built Rust"
else echo "skip Rust (no cargo)"; fi
if command -v go >/dev/null; then
  (cd go && go build -buildmode=c-shared -o ../out/libkernel_go.so kernel.go) && echo "built Go"
else echo "skip Go (no go)"; fi
