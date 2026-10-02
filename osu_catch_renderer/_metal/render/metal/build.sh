#!/bin/zsh
# Rebuild libr3dmetal.dylib from src/ (Apple Silicon, Xcode command-line tools).
# The committed dylib was built with Apple Swift 6.4, target arm64-apple-macosx26.0.
set -e
cd "$(dirname "$0")"
swiftc -O -emit-library -o libr3dmetal.dylib \
  src/r3dmetal.swift src/shaders.swift src/yuv.swift src/flash.swift src/death.swift \
  src/hpbar.swift src/hpinline.swift src/pill.swift \
  -framework Metal -framework Foundation
echo "built: $(ls -la libr3dmetal.dylib | awk '{print $5}') bytes"
