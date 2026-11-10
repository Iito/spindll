# SpindllKit

Swift package for spindll's `Scorer` on iOS: the next-token distribution after a context, candidate scoring, and short greedy extensions. It's built for keyboards, which re-score a growing context on every keystroke.

## Build

The package wraps `SpindllFFI.xcframework`, the static library from the `spindll-ffi` crate. Build it first:

```sh
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
scripts/build-ios-xcframework.sh          # keyboard build: no Metal, no weight repacking
scripts/build-ios-xcframework.sh --metal  # containing-app build, with the GPU
```

Then add `ios/SpindllKit` to the app as a local package.

## Use

```swift
import SpindllKit

let scorer = try Scorer(modelPath: path)          // keyboard defaults
try scorer.setContext("I would like to go to the")
let ends = try scorer.tokenize(" ,.")             // tokens that end a word
let scores = try scorer.score([" park", " moon"], terminators: ends)
```

Each `setContext` only decodes the tokens past the prefix it shares with the previous context. Calls run one at a time, so a scorer can be shared across threads. Errors throw `SpindllError` with spindll's message.

## Memory

A keyboard extension gets about 76 MB on an iPhone 15, so the default build leaves llama.cpp's Metal backend out. On an iPhone 16 Pro Max, Metal's setup costs 55 MB as soon as the backend starts, even with every layer on the CPU. Without it, Gemma 3 270M Q4_0 runs in about 32 MB. Weights are memory-mapped, and clean file-backed pages don't count toward the footprint iOS enforces.
