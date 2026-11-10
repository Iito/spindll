// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

import Foundation
import SpindllFFI

/// An error from spindll, with its message.
public struct SpindllError: Error, CustomStringConvertible {
    public let description: String

    static func last() -> SpindllError {
        SpindllError(description: spindll_last_error().map { String(cString: $0) } ?? "unknown spindll error")
    }
}

/// Scores text with a small language model: the next-token distribution after
/// a context, candidate continuations, and short greedy extensions. Built for
/// keyboards, which re-score a growing context on every keystroke.
///
/// The context is kept between calls, so a new context only pays for the
/// tokens past the prefix it shares with the previous one. A scorer runs its
/// calls one at a time, so it can be shared across threads.
public final class Scorer: @unchecked Sendable {
    public struct Config: Sendable {
        /// KV cache size in tokens: the context plus every candidate's tokens.
        public var contextSize: UInt32
        /// Tokens per decode. Also caps the logits buffer at this many rows.
        public var batchSize: UInt32
        /// Candidates scored side by side.
        public var parallel: UInt32
        /// Context tokens kept: the start token and the most recent ones.
        public var maxContextTokens: UInt32
        /// Candidates longer than this, in tokens, score -infinity.
        public var maxCandidateTokens: UInt32
        public var threads: Int32
        /// Layers on the GPU. 0 keeps everything on the CPU.
        public var gpuLayers: UInt32

        /// Defaults for a keyboard extension running Gemma 3 270M.
        public init() {
            var raw = SpindllScorerConfig()
            spindll_scorer_config_default(&raw)
            contextSize = raw.n_ctx
            batchSize = raw.n_batch
            parallel = raw.parallel
            maxContextTokens = raw.max_context_tokens
            maxCandidateTokens = raw.max_candidate_tokens
            threads = raw.threads
            gpuLayers = raw.gpu_layers
        }

        var raw: SpindllScorerConfig {
            SpindllScorerConfig(
                n_ctx: contextSize,
                n_batch: batchSize,
                parallel: parallel,
                max_context_tokens: maxContextTokens,
                max_candidate_tokens: maxCandidateTokens,
                threads: threads,
                gpu_layers: gpuLayers
            )
        }
    }

    private let handle: OpaquePointer
    /// Number of token ids; token flag arrays hold this many entries.
    public let vocabularySize: Int
    public let bos: Int32
    public let eos: Int32

    /// Loads a GGUF model.
    public init(modelPath: String, config: Config = Config()) throws {
        var raw = config.raw
        guard let handle = spindll_scorer_load(modelPath, &raw) else { throw SpindllError.last() }
        self.handle = handle
        vocabularySize = Int(spindll_scorer_n_vocab(handle))
        bos = spindll_scorer_bos(handle)
        eos = spindll_scorer_eos(handle)
    }

    deinit {
        spindll_scorer_free(handle)
    }

    /// The text of `token`.
    public func piece(_ token: Int32) throws -> String {
        var buffer = [CChar](repeating: 0, count: 64)
        var n = spindll_scorer_piece(handle, token, &buffer, Int32(buffer.count))
        guard n >= 0 else { throw SpindllError.last() }
        if Int(n) > buffer.count {
            buffer = [CChar](repeating: 0, count: Int(n))
            n = spindll_scorer_piece(handle, token, &buffer, n)
            guard n >= 0 else { throw SpindllError.last() }
        }
        return String(decoding: buffer[0..<Int(n)].map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// The text of every token, indexed by token id, for building token flags.
    public func pieces() throws -> [String] {
        try (0..<Int32(vocabularySize)).map(piece)
    }

    public func tokenize(_ text: String, addBOS: Bool = false) throws -> [Int32] {
        var tokens = [Int32](repeating: 0, count: text.utf8.count + 8)
        var n = spindll_scorer_tokenize(handle, text, addBOS, &tokens, Int32(tokens.count))
        guard n >= 0 else { throw SpindllError.last() }
        if Int(n) > tokens.count {
            tokens = [Int32](repeating: 0, count: Int(n))
            n = spindll_scorer_tokenize(handle, text, addBOS, &tokens, n)
            guard n >= 0 else { throw SpindllError.last() }
        }
        return Array(tokens[0..<Int(n)])
    }

    /// Makes `text` the context for later calls.
    public func setContext(_ text: String) throws {
        guard spindll_scorer_set_context(handle, text) == 0 else { throw SpindllError.last() }
    }

    public func setContext(tokens: [Int32]) throws {
        guard spindll_scorer_set_context_tokens(handle, tokens, Int32(tokens.count)) == 0 else {
            throw SpindllError.last()
        }
    }

    /// Log-probability of each token coming right after the context.
    public func nextLogprobs(_ tokens: [Int32]) throws -> [Float] {
        var logprobs = [Float](repeating: 0, count: tokens.count)
        guard spindll_scorer_next_logprobs(handle, tokens, Int32(tokens.count), &logprobs) == 0 else {
            throw SpindllError.last()
        }
        return logprobs
    }

    /// The `k` most likely next tokens, best first, among `allowed` if given
    /// (one flag per token id).
    public func topNext(_ k: Int, allowed: [Bool]? = nil) throws -> [(token: Int32, logprob: Float)] {
        var tokens = [Int32](repeating: 0, count: k)
        var logprobs = [Float](repeating: 0, count: k)
        let n: Int32
        if let allowed {
            precondition(allowed.count == vocabularySize, "allowed needs one flag per token id")
            n = allowed.withUnsafeBufferPointer {
                spindll_scorer_top_next(handle, Int32(k), $0.baseAddress, &tokens, &logprobs)
            }
        } else {
            n = spindll_scorer_top_next(handle, Int32(k), nil, &tokens, &logprobs)
        }
        guard n >= 0 else { throw SpindllError.last() }
        return (0..<Int(n)).map { (tokens[$0], logprobs[$0]) }
    }

    /// Log-probability of each text coming next, then one of `terminators`
    /// (e.g. the tokens that end a word). With no terminators, just the text.
    public func score(_ texts: [String], terminators: [Int32] = []) throws -> [Float] {
        var scores = [Float](repeating: 0, count: texts.count)
        let rc = withCStrings(texts) { pointers in
            spindll_scorer_score(
                handle, pointers, Int32(texts.count), terminators, Int32(terminators.count), &scores
            )
        }
        guard rc == 0 else { throw SpindllError.last() }
        return scores
    }

    /// Extends each start token greedily, at most `maxSteps` tokens, while the
    /// most likely next token is flagged in `keepGoing` (one flag per token
    /// id). Each run begins with its start token.
    public func extendGreedy(_ starts: [Int32], maxSteps: Int, keepGoing: [Bool]) throws -> [[Int32]] {
        precondition(keepGoing.count == vocabularySize, "keepGoing needs one flag per token id")
        let stride = maxSteps + 1
        var tokens = [Int32](repeating: 0, count: starts.count * stride)
        var lengths = [Int32](repeating: 0, count: starts.count)
        let rc = keepGoing.withUnsafeBufferPointer {
            spindll_scorer_extend_greedy(
                handle, starts, Int32(starts.count), Int32(maxSteps), $0.baseAddress, &tokens, &lengths
            )
        }
        guard rc == 0 else { throw SpindllError.last() }
        return (0..<starts.count).map { Array(tokens[$0 * stride ..< $0 * stride + Int(lengths[$0])]) }
    }
}

/// Calls `body` with a C array of NUL-terminated copies of `strings`.
private func withCStrings<R>(_ strings: [String], _ body: (UnsafePointer<UnsafePointer<CChar>?>?) -> R) -> R {
    let copies = strings.map { strdup($0) }
    defer { copies.forEach { free($0) } }
    let pointers = copies.map { $0.map { UnsafePointer($0) } }
    return pointers.withUnsafeBufferPointer { body($0.baseAddress) }
}
