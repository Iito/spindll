import Foundation
import CryptoKit
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import Tokenizers

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

// Thread-safe box for bridging async results to DispatchSemaphore.
// @unchecked Sendable is intentional: the semaphore guarantees happens-before.
private final class Box<T>: @unchecked Sendable {
    var value: T?
}

// ---------------------------------------------------------------------------
// PromptCache — in-memory LRU cache of KV state keyed by token sequence.
//
// On a cache *prefix* match the saved KV state is restored, trimmed back to the
// longest token prefix it shares with the new prompt, and only the tokens that
// are new since then are prefilled — so a follow-up turn in a chat (system +
// turn1 + turn2 + …) costs O(new tokens) of prefill instead of O(whole prompt).
// On an exact match (or when the new prompt is itself a prefix of the cached
// one) just the last prompt token is re-processed, so TTFT ≈ a single decode
// step.
//
// Each stored state is re-snapshotted *after* generation, so it covers the
// prompt plus the tokens generated that turn — that is exactly the prefix the
// next turn's prompt extends.
//
// Snapshots are quantized to keep the in-memory footprint small. The freshest
// entry — the one the active conversation keeps extending — is kept at the
// higher precision (`highBits`); older entries are demoted to `lowBits` the
// moment a newer snapshot supersedes them. So the hot path always reuses a
// near-lossless state while cold entries cost a quarter of an f16 state. This
// mirrors how macOS keeps active pages uncompressed and compresses inactive
// ones.
//
// Access is serialised by ModelContainer's SerialAccessContainer actor, so no
// additional lock is needed.
// ---------------------------------------------------------------------------

private let kvGroupSize = 64
private let highBits = 8     // freshest snapshot — near-lossless
private let lowBits  = 4     // demoted (cold) snapshot — ¼ of f16

/// Length of the longest common prefix of two token-id sequences.
private func commonPrefixLength(_ a: [Int32], _ b: [Int32]) -> Int {
    let n = min(a.count, b.count)
    var i = 0
    while i < n && a[i] == b[i] { i += 1 }
    return i
}

private final class PromptCacheEntry {
    /// Token sequence this KV state corresponds to — prompt tokens plus any
    /// tokens generated after it, so a later turn can prefix off this entry.
    var tokenIds: [Int32]
    /// KV state covering exactly `tokenIds`: every layer has `offset == tokenIds.count`.
    var kvCache: [any KVCache]
    /// Bit width of the quantized attention layers (`highBits` or `lowBits`).
    var bits: Int
    /// A copy of this state has already been written to the on-disk tier — so
    /// it isn't re-written when the entry is later demoted or evicted.
    var spilled = false

    init(tokenIds: [Int32], kvCache: [any KVCache], bits: Int) {
        self.tokenIds = tokenIds
        self.kvCache  = kvCache
        self.bits     = bits
    }
}

/// Result of a prompt-cache lookup: the matched entry and how many leading
/// tokens it shares with the requested prompt.
private struct PromptCacheHit {
    let entry: PromptCacheEntry
    let prefixLength: Int
}

/// Re-quantize an entry's attention layers down to `lowBits`. Only ever called
/// on entries that are no longer the most-recently-used, to shrink the in-memory
/// footprint. Mamba/SSM layers (already independent deep copies) are left as-is.
private func demote(_ entry: PromptCacheEntry) {
    guard entry.bits != lowBits else { return }
    entry.kvCache = entry.kvCache.map { layer -> any KVCache in
        guard let q = layer as? QuantizedKVCache, q.bits != lowBits else { return layer }
        return q.toUnquantized().toQuantized(groupSize: kvGroupSize, bits: lowBits)
    }
    entry.bits = lowBits
}

// ---------------------------------------------------------------------------
// MlxDiskCache — on-disk LRU tier behind PromptCache.
//
// Evicted PromptCache entries are written here as safetensors (via
// mlx-swift-lm's savePromptCache); a RAM miss checks here for a usable token
// prefix before falling back to a cold prefill, so prompt prefixes survive
// process restarts. Shared by every loaded model — entries are namespaced by a
// digest of the model's config.json, so re-downloading a model invalidates them.
// Set SPINDLL_MLX_DISK_CACHE=0 to disable.
// ---------------------------------------------------------------------------

private func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private struct DiskCacheEntry: Codable {
    var file: String          // "<modelDigest>_<tokenDigest>.safetensors"
    var modelDigest: String
    var tokenIds: [Int32]
    var bits: Int             // quantization width of the attention layers
    var sizeBytes: Int64
    var lastUsed: Double      // for LRU ordering
}

private final class MlxDiskCache {
    private let dir: URL
    private let maxBytes: Int64
    private let enabled: Bool
    private let lock = NSLock()
    private var index: [String: DiskCacheEntry] = [:]

    init(maxBytes: Int64 = 2 * 1024 * 1024 * 1024) {
        self.maxBytes = maxBytes
        self.enabled = ProcessInfo.processInfo.environment["SPINDLL_MLX_DISK_CACHE"] != "0"
        let home = ProcessInfo.processInfo.environment["HOME"] ?? NSHomeDirectory()
        self.dir = URL(fileURLWithPath: home)
            .appendingPathComponent(".spindll").appendingPathComponent("cache").appendingPathComponent("mlx")
        guard enabled else { return }
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let data = try? Data(contentsOf: dir.appendingPathComponent("index.json")),
           let list = try? JSONDecoder().decode([DiskCacheEntry].self, from: data) {
            index = Dictionary(list.map { ($0.file, $0) }, uniquingKeysWith: { a, _ in a })
        }
    }

    private func saveIndexLocked() {
        if let data = try? JSONEncoder().encode(Array(index.values)) {
            try? data.write(to: dir.appendingPathComponent("index.json"), options: .atomic)
        }
    }

    private func totalBytesLocked() -> Int64 { index.values.reduce(0) { $0 + $1.sizeBytes } }

    private func evictLocked() {
        while totalBytesLocked() > maxBytes,
              let lru = index.values.min(by: { $0.lastUsed < $1.lastUsed }) {
            index[lru.file] = nil
            try? FileManager.default.removeItem(at: dir.appendingPathComponent(lru.file))
        }
    }

    /// Find this model's disk entry sharing the longest token prefix with
    /// `tokenIds` (≥ `minPrefix`), load it, refresh its LRU stamp, and return
    /// its reconstructed KV state. `nil` on no match or a load failure.
    func lookup(modelDigest: String, tokenIds: [Int32], minPrefix: Int)
        -> (tokenIds: [Int32], kvCache: [any KVCache], bits: Int)? {
        guard enabled else { return nil }
        lock.lock()
        var best: DiskCacheEntry? = nil
        var bestLen = 0
        for e in index.values where e.modelDigest == modelDigest {
            let l = commonPrefixLength(e.tokenIds, tokenIds)
            if l > bestLen { bestLen = l; best = e }
        }
        guard var entry = best, bestLen >= minPrefix else { lock.unlock(); return nil }
        let url = dir.appendingPathComponent(entry.file)
        lock.unlock()

        let loaded: [KVCache]
        do { loaded = try loadPromptCache(url: url).0 }
        catch {
            fputs("[mlx-bridge] disk cache: load failed (\(entry.file)): \(error)\n", stderr)
            lock.lock(); index[entry.file] = nil; saveIndexLocked(); lock.unlock()
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        // loadPromptCache rebuilds QuantizedKVCache layers at the default 8-bit
        // width — mlx-swift-lm doesn't restore bits/groupSize from metaState — so
        // re-bind the loaded arrays into caches with the width we recorded.
        let kvCache: [any KVCache] = entry.bits == 8 ? loaded : loaded.map { layer in
            guard let q = layer as? QuantizedKVCache, q.bits != entry.bits else { return layer }
            let nq = QuantizedKVCache(groupSize: q.groupSize, bits: entry.bits)
            nq.state = q.state
            nq.offset = q.offset
            return nq
        }
        lock.lock()
        entry.lastUsed = Date().timeIntervalSinceReferenceDate
        index[entry.file] = entry
        saveIndexLocked()
        lock.unlock()
        return (entry.tokenIds, kvCache, entry.bits)
    }

    /// Write an evicted RAM entry to disk, dropping LRU disk entries if over budget.
    func put(modelDigest: String, tokenIds: [Int32], kvCache: [any KVCache], bits: Int) {
        guard enabled, !tokenIds.isEmpty else { return }
        var tokenData = Data(capacity: tokenIds.count * 4)
        for t in tokenIds { withUnsafeBytes(of: t.littleEndian) { tokenData.append(contentsOf: $0) } }
        let file = "\(modelDigest)_\(sha256Hex(tokenData)).safetensors"
        let url = dir.appendingPathComponent(file)
        do { try savePromptCache(url: url, cache: kvCache, metadata: [:]) }
        catch {
            fputs("[mlx-bridge] disk cache: save failed (\(file)): \(error)\n", stderr)
            try? FileManager.default.removeItem(at: url)
            return
        }
        let size = Int64((try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
        lock.lock()
        index[file] = DiskCacheEntry(file: file, modelDigest: modelDigest, tokenIds: tokenIds,
                                     bits: bits, sizeBytes: size, lastUsed: Date().timeIntervalSinceReferenceDate)
        evictLocked()
        if index[file] == nil { try? FileManager.default.removeItem(at: url) }  // entry alone > budget
        saveIndexLocked()
        lock.unlock()
    }
}

private let sharedMlxDiskCache = MlxDiskCache()

private final class PromptCache {
    private var entries: [PromptCacheEntry] = []
    private let maxSize: Int
    /// Don't reuse a state that shares only a handful of tokens — the prefill we
    /// would save is dwarfed by the copy + trim overhead.
    private let minPrefix: Int
    /// Only touch the on-disk tier for prompts at least this long — for short
    /// prompts, reading an ~tens-of-MB safetensors back costs more than just
    /// re-prefilling. The RAM tier is used regardless of length.
    private let diskMinTokens = 512
    /// Model identity (digest of config.json) — namespaces this model's entries
    /// in the shared on-disk tier.
    private let modelDigest: String

    init(modelDigest: String, maxSize: Int = 2, minPrefix: Int = 16) {
        self.modelDigest = modelDigest
        self.maxSize = maxSize
        self.minPrefix = minPrefix
    }

    /// RAM lookup: the entry sharing the longest token prefix with `tokenIds`
    /// (≥ `minPrefix`), promoted to the front of the LRU list.
    private func lookupRam(_ tokenIds: [Int32]) -> PromptCacheHit? {
        var bestIdx: Int? = nil
        var bestLen = 0
        for (i, e) in entries.enumerated() {
            let l = commonPrefixLength(e.tokenIds, tokenIds)
            if l > bestLen { bestLen = l; bestIdx = i }
        }
        guard let idx = bestIdx, bestLen >= minPrefix else { return nil }
        let entry = entries.remove(at: idx)
        entries.insert(entry, at: 0)
        return PromptCacheHit(entry: entry, prefixLength: bestLen)
    }

    /// Look up `tokenIds`, checking RAM first and then the on-disk tier; a disk
    /// hit is loaded back into RAM. `nil` if neither has a usable prefix.
    func lookup(tokenIds: [Int32]) -> PromptCacheHit? {
        guard !tokenIds.isEmpty else { return nil }
        if let hit = lookupRam(tokenIds) { return hit }
        guard tokenIds.count >= diskMinTokens,
              let d = sharedMlxDiskCache.lookup(
                modelDigest: modelDigest, tokenIds: tokenIds, minPrefix: minPrefix)
        else { return nil }
        let entry = PromptCacheEntry(tokenIds: d.tokenIds, kvCache: d.kvCache, bits: d.bits)
        entry.spilled = true   // it came from disk; no need to write it back
        entries.removeAll { commonPrefixLength($0.tokenIds, entry.tokenIds) == $0.tokenIds.count }
        entries.insert(entry, at: 0)
        evictExcess()
        return PromptCacheHit(entry: entry, prefixLength: commonPrefixLength(entry.tokenIds, tokenIds))
    }

    /// Insert (or replace) the entry for `tokenIds`. Any existing entry whose
    /// token sequence is a prefix of the new one is dropped (the new entry
    /// supersedes it). Surviving older entries spill a near-lossless copy to the
    /// on-disk tier and are then demoted to `lowBits` in RAM. The LRU entry, if
    /// the cache is over capacity, is dropped from RAM (already on disk).
    func save(tokenIds: [Int32], kvCache: [any KVCache], bits: Int) {
        guard !tokenIds.isEmpty else { return }
        entries.removeAll { commonPrefixLength($0.tokenIds, tokenIds) == $0.tokenIds.count }
        for e in entries {
            spillToDisk(e)   // while still highBits — disk entries stay near-lossless
            demote(e)
        }
        entries.insert(PromptCacheEntry(tokenIds: tokenIds, kvCache: kvCache, bits: bits), at: 0)
        evictExcess()
    }

    private func evictExcess() {
        while entries.count > maxSize, let evicted = entries.popLast() {
            spillToDisk(evicted)   // no-op if already spilled or already demoted
        }
    }

    /// Write `entry`'s state to the on-disk tier once, while it's still at the
    /// higher precision and only if it's long enough to be worth the I/O.
    private func spillToDisk(_ entry: PromptCacheEntry) {
        guard !entry.spilled, entry.bits == highBits, entry.tokenIds.count >= diskMinTokens else { return }
        sharedMlxDiskCache.put(modelDigest: modelDigest,
                               tokenIds: entry.tokenIds, kvCache: entry.kvCache, bits: entry.bits)
        entry.spilled = true
    }
}

// Retains the ModelContainer and prompt KV cache across FFI calls.
private final class ModelState: @unchecked Sendable {
    let container: ModelContainer
    let promptCache: PromptCache
    init(_ container: ModelContainer, modelDigest: String) {
        self.container = container
        self.promptCache = PromptCache(modelDigest: modelDigest)
    }
}

// ---------------------------------------------------------------------------
// mlx_model_load
// ---------------------------------------------------------------------------

/// Load a model from a local MLX-format directory.
/// Blocks the calling thread until the model is fully resident in memory.
/// Returns NULL on failure; caller must free with mlx_model_free.
@_cdecl("mlx_model_load")
public func mlxModelLoad(_ path: UnsafePointer<CChar>?) -> UnsafeMutableRawPointer? {
    guard let path else { return nil }
    let modelPath = String(cString: path)
    let url = URL(fileURLWithPath: modelPath)

    let sema = DispatchSemaphore(value: 0)
    let box = Box<ModelState>()

    // Model identity for the on-disk prompt cache: digest of config.json (the
    // same artefact the Rust side uses as the MLX model version), falling back
    // to the path if it can't be read.
    let modelDigest: String
    if let cfg = try? Data(contentsOf: url.appendingPathComponent("config.json")) {
        modelDigest = sha256Hex(cfg)
    } else {
        modelDigest = sha256Hex(Data(modelPath.utf8))
    }

    Task {
        do {
            let container = try await LLMModelFactory.shared.loadContainer(
                from: url,
                using: #huggingFaceTokenizerLoader()
            )
            box.value = ModelState(container, modelDigest: modelDigest)
        } catch {
            // box.value stays nil; Rust side receives NULL
        }
        sema.signal()
    }
    sema.wait()

    guard let state = box.value else { return nil }
    return Unmanaged.passRetained(state).toOpaque()
}

// ---------------------------------------------------------------------------
// mlx_model_free
// ---------------------------------------------------------------------------

/// Release the model and free memory. Safe to call with NULL.
@_cdecl("mlx_model_free")
public func mlxModelFree(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    Unmanaged<ModelState>.fromOpaque(handle).release()
}

// ---------------------------------------------------------------------------
// mlx_generate
// ---------------------------------------------------------------------------

/// Blocking generation: drives a sync `TokenIterator` inside
/// `ModelContainer.perform(nonSendable:)` and calls `callback` per
/// detokenizer chunk. Returns the number of tokens generated on
/// success, -1 on error.
@_cdecl("mlx_generate")
public func mlxGenerate(
    _ handle:      UnsafeMutableRawPointer?,
    _ prompt:      UnsafePointer<CChar>?,
    _ maxTokens:   UInt32,
    _ temperature: Float,
    _ topP:        Float,
    _ seed:        UInt32,
    _ callback:    @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Int32,
    _ callbackCtx: UnsafeMutableRawPointer?
) -> Int32 {
    guard let handle, let prompt else { return -1 }
    let state = Unmanaged<ModelState>.fromOpaque(handle).takeUnretainedValue()
    let promptStr = String(cString: prompt)

    let sema = DispatchSemaphore(value: 0)
    let box = Box<Int32>()

    Task {
        do {
            let params = GenerateParameters(
                maxTokens: Int(maxTokens),
                temperature: temperature,
                topP: topP
            )

            // Tokenise the already chat-template-formatted prompt directly.
            // Using UserInput(prompt:) + container.prepare() would re-apply the
            // chat template (wrapping the formatted string inside another user
            // turn) and costs an extra actor entry. Instead, encode the string
            // once and construct LMInput directly inside a single perform call.
            var generated: Int32 = 0
            try await state.container.perform(nonSendable: promptStr) { context, prompt in
                let tokenIds = context.tokenizer.encode(text: prompt, addSpecialTokens: false)
                let lmInput = LMInput(tokens: MLXArray(tokenIds.map { Int32($0) }))
                var iterator = try TokenIterator(
                    input: lmInput,
                    model: context.model,
                    parameters: params
                )
                var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)

                var stopTokenIds: Set<Int> = []
                if let eos = context.tokenizer.eosTokenId { stopTokenIds.insert(eos) }
                if let unk = context.tokenizer.unknownTokenId { stopTokenIds.insert(unk) }
                for id in context.configuration.eosTokenIds { stopTokenIds.insert(id) }
                // Resolve string-form stop tokens to ids — Gemma3 (`<end_of_turn>`),
                // Phi (`<|end|>`), SmolLM (`<turn|>`) etc. live here, not in eosTokenIds.
                for token in context.configuration.extraEOSTokens {
                    if let id = context.tokenizer.convertTokenToId(token) {
                        stopTokenIds.insert(id)
                    }
                }

                var cancelled = false
                while let tokenId = iterator.next() {
                    if stopTokenIds.contains(tokenId) { break }
                    generated += 1
                    detokenizer.append(token: tokenId)
                    if let chunk = detokenizer.next() {
                        let shouldContinue = chunk.withCString { ptr in
                            callback(ptr, callbackCtx)
                        }
                        // Caller dropped -- break instead of decoding to EOS
                        // and discarding chunks.
                        if shouldContinue == 0 {
                            cancelled = true
                            break
                        }
                    }
                }

                // Flush any partial-UTF-8 bytes the detokenizer was holding
                // back (e.g. when the loop hits maxTokens mid-codepoint).
                if !cancelled, let chunk = detokenizer.next() {
                    _ = chunk.withCString { ptr in callback(ptr, callbackCtx) }
                }

                // Wait for in-flight async evaluations to finish before the
                // perform closure tears down — mirrors upstream's sync loop
                // in `runSynchronousGenerationLoop` (Evaluate.swift:1134).
                Stream().synchronize()
            }
            box.value = generated
        } catch {
            fputs("[mlx-bridge] mlx_generate error: \(error)\n", stderr)
            box.value = -1
        }
        sema.signal()
    }
    sema.wait()

    return box.value ?? -1
}

// ---------------------------------------------------------------------------
// ChatML fallback
// ---------------------------------------------------------------------------

/// Apply the chat template, falling back to ChatML if the tokenizer has
/// no built-in template.  Returns token IDs.
///
/// Models whose `tokenizer_config.json` omits the `chat_template` field
/// (common with multimodal models like Qwen-VL) will hit the fallback
/// path: messages are formatted in ChatML and encoded directly.
private func applyChatTemplateWithFallback(
    tokenizer: any MLXLMCommon.Tokenizer,
    messages: [[String: String]]
) throws -> [Int] {
    do {
        return try tokenizer.applyChatTemplate(messages: messages)
    } catch is MLXLMCommon.TokenizerError {
        fputs("[mlx-bridge] model has no chat template, using ChatML fallback\n", stderr)
        var chatML = ""
        for msg in messages {
            if let role = msg["role"], let content = msg["content"] {
                chatML += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
            }
        }
        chatML += "<|im_start|>assistant\n"
        return tokenizer.encode(text: chatML, addSpecialTokens: false)
    }
}

// ---------------------------------------------------------------------------
// Deep-copy cache layer
// ---------------------------------------------------------------------------

/// Create an independent deep copy of a KV/SSM cache layer.
///
/// `.copy()` in mlx-swift-lm uses `array[.ellipsis]` which is a slice view —
/// the new arrays share the underlying buffer and get corrupted when the
/// original is mutated during generation.
///
/// For KVCacheSimple we use `toQuantized()` (handled at call-site).
/// For Mamba/SSM caches and anything else, we force an independent allocation
/// via an arithmetic identity (`* 1`) + `eval()`.
private func deepCopyCache(_ layer: any KVCache) -> any KVCache {
    var copied = layer.copy()
    if copied is KVCacheSimple { return copied }
    // .copy() uses array[.ellipsis] which is a view sharing the underlying
    // buffer. Force independent allocations via an arithmetic identity.
    let detached = copied.state.map { $0 * 1 }
    MLX.eval(detached)
    copied.state = detached
    return copied
}

/// Take an independent snapshot of a live KV cache for the prompt cache.
///
/// Attention layers are quantized to `bits` (`trim()` just decrements the offset
/// so prefix reuse is unaffected by quantization); Mamba/SSM layers are deep-
/// copied as-is. Works whether the live cache is f16 (`KVCacheSimple`, on a
/// miss) or already quantized at some other width (`QuantizedKVCache`, on a hit
/// off a previous snapshot). The result is independent — safe to retain while
/// generation keeps mutating the live cache.
private func snapshotCache(_ cache: [any KVCache], bits: Int) -> [any KVCache] {
    cache.map { layer in
        if let simple = layer as? KVCacheSimple {
            return simple.toQuantized(groupSize: kvGroupSize, bits: bits)
        }
        if let q = layer as? QuantizedKVCache, q.bits != bits {
            return q.toUnquantized().toQuantized(groupSize: kvGroupSize, bits: bits)
        }
        return deepCopyCache(layer)
    }
}

// ---------------------------------------------------------------------------
// mlx_chat_generate
// ---------------------------------------------------------------------------

/// Apply the model's chat template and generate tokens in a single actor entry.
///
/// `messagesJson` is a UTF-8 JSON array of `{"role": ..., "content": ...}`
/// objects. The tokeniser's Jinja chat template is applied to obtain token
/// IDs which are fed directly to `TokenIterator` — no decode → encode
/// round-trip. Returns the number of tokens generated on success, -1 on error.
@_cdecl("mlx_chat_generate")
public func mlxChatGenerate(
    _ handle:      UnsafeMutableRawPointer?,
    _ messagesJson: UnsafePointer<CChar>?,
    _ maxTokens:   UInt32,
    _ temperature: Float,
    _ topP:        Float,
    _ seed:        UInt32,
    _ callback:    @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Int32,
    _ callbackCtx: UnsafeMutableRawPointer?
) -> Int32 {
    guard let handle, let messagesJson else { return -1 }
    let state = Unmanaged<ModelState>.fromOpaque(handle).takeUnretainedValue()
    let json = String(cString: messagesJson)

    guard
        let data = json.data(using: .utf8),
        let parsed = try? JSONSerialization.jsonObject(with: data),
        let messages = parsed as? [[String: String]]
    else { return -1 }

    let sema = DispatchSemaphore(value: 0)
    let box = Box<Int32>()

    Task {
        do {
            let params = GenerateParameters(
                maxTokens: Int(maxTokens),
                temperature: temperature,
                topP: topP
            )

            var generated: Int32 = 0
            try await state.container.perform(nonSendable: messages) { context, msgs in
                // Apply the chat template once → token IDs. No decode → encode round-trip.
                let rawIds  = try applyChatTemplateWithFallback(
                    tokenizer: context.tokenizer, messages: msgs)
                let tokenIds = rawIds.map { Int32($0) }

                // --- Prompt KV cache (prefix reuse) ---
                // Look for a cached state sharing a token prefix with this
                // prompt. On a hit we restore that state, trim it back to the
                // shared-prefix length, and prefill only the new tokens; on a
                // miss we prefill the whole prompt into a fresh cache. Either
                // way the live cache is re-snapshotted after generation (so it
                // covers prompt + generated tokens) for the next prefix lookup.
                let liveCache: [any KVCache]
                var iterator: TokenIterator
                let snapshotBits: Int   // precision to re-snapshot the live cache at

                if let hit = state.promptCache.lookup(tokenIds: tokenIds),
                   canTrimPromptCache(hit.entry.kvCache) {
                    let restored = hit.entry.kvCache.map { deepCopyCache($0) }
                    let cachedLen = hit.entry.tokenIds.count
                    let prefixLen = hit.prefixLength      // 1 ≤ prefixLen ≤ tokenIds.count, ≤ cachedLen

                    // Roll the restored state back to the shared prefix.
                    if cachedLen > prefixLen {
                        trimPromptCache(restored, numTokens: cachedLen - prefixLen)
                    }

                    let seedTokens: [Int32]
                    if prefixLen < tokenIds.count {
                        seedTokens = Array(tokenIds[prefixLen...])
                    } else {
                        // Whole prompt already cached — re-process just the last
                        // token so TokenIterator has logits to sample from.
                        trimPromptCache(restored, numTokens: 1)
                        seedTokens = [tokenIds[tokenIds.count - 1]]
                    }

                    liveCache = restored
                    // Keep the precision we restored at — re-quantizing a coarse
                    // state to a finer width recovers no information.
                    snapshotBits = hit.entry.bits
                    iterator = try TokenIterator(
                        input: LMInput(tokens: MLXArray(seedTokens)),
                        model: context.model,
                        cache: liveCache,
                        parameters: params
                    )
                } else {
                    liveCache = makePromptCache(model: context.model, parameters: params)
                    // Miss: the live cache is f16 — snapshot it near-losslessly.
                    snapshotBits = highBits
                    iterator = try TokenIterator(
                        input: LMInput(tokens: MLXArray(tokenIds)),
                        model: context.model,
                        cache: liveCache,
                        parameters: params
                    )
                }

                var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)

                var stopTokenIds: Set<Int> = []
                if let eos = context.tokenizer.eosTokenId { stopTokenIds.insert(eos) }
                if let unk = context.tokenizer.unknownTokenId { stopTokenIds.insert(unk) }
                for id in context.configuration.eosTokenIds { stopTokenIds.insert(id) }
                for token in context.configuration.extraEOSTokens {
                    if let id = context.tokenizer.convertTokenToId(token) {
                        stopTokenIds.insert(id)
                    }
                }

                var cancelled = false
                var generatedTokenIds: [Int32] = []
                while let tokenId = iterator.next() {
                    // TokenIterator has appended `tokenId` to liveCache by now,
                    // so liveCache.offset == tokenIds.count + generatedTokenIds.count.
                    generatedTokenIds.append(Int32(tokenId))
                    if stopTokenIds.contains(tokenId) { break }
                    generated += 1
                    detokenizer.append(token: tokenId)
                    if let chunk = detokenizer.next() {
                        let shouldContinue = chunk.withCString { ptr in
                            callback(ptr, callbackCtx)
                        }
                        if shouldContinue == 0 {
                            cancelled = true
                            break
                        }
                    }
                }

                if !cancelled, let chunk = detokenizer.next() {
                    _ = chunk.withCString { ptr in callback(ptr, callbackCtx) }
                }

                // Snapshot the live cache — now covering prompt + generated
                // tokens — for the next turn's prefix lookup. Skip caches that
                // can't be trimmed (Mamba/SSM hybrids): they could only ever be
                // matched whole, which is never useful for prefix reuse.
                if canTrimPromptCache(liveCache) {
                    state.promptCache.save(
                        tokenIds: tokenIds + generatedTokenIds,
                        kvCache: snapshotCache(liveCache, bits: snapshotBits),
                        bits: snapshotBits)
                }

                Stream().synchronize()
            }
            box.value = generated
        } catch {
            fputs("[mlx-bridge] mlx_chat_generate error: \(error)\n", stderr)
            box.value = -1
        }
        sema.signal()
    }
    sema.wait()

    return box.value ?? -1
}

// ---------------------------------------------------------------------------
// mlx_apply_chat_template
// ---------------------------------------------------------------------------

/// Render a chat template using the model's tokenizer.
///
/// `messagesJson` is a UTF-8 JSON array of `{"role": ..., "content": ...}`
/// objects. The tokenizer's Jinja chat template is applied (with
/// `add_generation_prompt: true` so the assistant header is injected),
/// then decoded back to a UTF-8 string. Returns NULL on any failure;
/// otherwise the caller owns the returned pointer and must release it
/// via `mlx_free_string`.
@_cdecl("mlx_apply_chat_template")
public func mlxApplyChatTemplate(
    _ handle: UnsafeMutableRawPointer?,
    _ messagesJson: UnsafePointer<CChar>?
) -> UnsafePointer<CChar>? {
    guard let handle, let messagesJson else { return nil }
    let state = Unmanaged<ModelState>.fromOpaque(handle).takeUnretainedValue()
    let json = String(cString: messagesJson)

    // Decode JSON → [[String: String]] (the shape Tokenizers' applyChatTemplate expects).
    guard
        let data = json.data(using: .utf8),
        let parsed = try? JSONSerialization.jsonObject(with: data),
        let messages = parsed as? [[String: String]]
    else { return nil }

    let sema = DispatchSemaphore(value: 0)
    let box = Box<String>()

    Task {
        do {
            try await state.container.perform { context in
                // Tokenizers.applyChatTemplate returns token IDs; decode back
                // to a string so we can hand a rendered prompt to llama.cpp's
                // tokenizer (the MLX path then re-tokenizes via UserInput).
                // add_generation_prompt is on by default in swift-transformers.
                let ids = try applyChatTemplateWithFallback(
                    tokenizer: context.tokenizer, messages: messages)
                box.value = context.tokenizer.decode(tokenIds: ids)
            }
        } catch {
            fputs("[mlx-bridge] mlx_apply_chat_template error: \(error)\n", stderr)
            // box.value stays nil → NULL returned
        }
        sema.signal()
    }
    sema.wait()

    guard let rendered = box.value,
          let cstr = rendered.cString(using: .utf8)
    else { return nil }

    // Allocate a new buffer and copy. Caller owns it and must release via
    // mlx_free_string. We use UnsafeMutablePointer<CChar>.allocate(...) so
    // free() in Swift matches deallocate() — same allocator end-to-end.
    let buf = UnsafeMutablePointer<CChar>.allocate(capacity: cstr.count)
    buf.initialize(from: cstr, count: cstr.count)
    return UnsafePointer(buf)
}

// ---------------------------------------------------------------------------
// mlx_free_string
// ---------------------------------------------------------------------------

/// Release a string previously returned by `mlx_apply_chat_template`.
@_cdecl("mlx_free_string")
public func mlxFreeString(_ s: UnsafePointer<CChar>?) {
    guard let s else { return }
    let mutable = UnsafeMutablePointer(mutating: s)
    // Length includes the NUL terminator.
    let len = strlen(s) + 1
    mutable.deinitialize(count: len)
    mutable.deallocate()
}

// ---------------------------------------------------------------------------
// mlx_embed
// ---------------------------------------------------------------------------

@_cdecl("mlx_embed")
public func mlxEmbed(
    _ handle:  UnsafeMutableRawPointer?,
    _ text:    UnsafePointer<CChar>?,
    _ outData: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outLen:  UnsafeMutablePointer<Int32>?
) -> Int32 {
    guard let handle, let text, let outData, let outLen else { return -1 }
    let state = Unmanaged<ModelState>.fromOpaque(handle).takeUnretainedValue()
    let inputText = String(cString: text)

    let sema = DispatchSemaphore(value: 0)
    let box = Box<(UnsafeMutablePointer<Float>, Int32, Int32)>()

    // NOTE: This path produces *heuristic* embeddings from a generative LM by
    // mean-pooling the input embedding rows for the tokenised text. It is NOT
    // a true pseudo-inverse hidden-state recovery — the previous
    // `matmul(logits, W_embed)` formula did not compute one either. Use a
    // dedicated embedding model (e.g. sentence-transformers MLX) for any
    // task where vector quality matters.
    Task {
        do {
            try await state.container.perform(nonSendable: inputText) { context, txt in
                let tokenIds = context.tokenizer.encode(text: txt)
                guard !tokenIds.isEmpty else { return }

                let allParams = context.model.parameters().flattened()

                let embedKeys = ["embed_tokens", "wte", "word_embeddings"]
                guard let prefix = embedKeys.lazy.compactMap({ name -> String? in
                    allParams.first(where: { $0.0.contains(name) && $0.0.hasSuffix(".weight") })
                        .map { String($0.0.dropLast(".weight".count)) }
                }).first else { return }

                guard let embedWeight = allParams.first(where: { $0.0 == prefix + ".weight" })?.1
                else { return }
                let embedScales = allParams.first(where: { $0.0 == prefix + ".scales" })?.1
                let embedBiases = allParams.first(where: { $0.0 == prefix + ".biases" })?.1

                let fullWeight: MLXArray
                if let embedScales {
                    fullWeight = dequantized(
                        embedWeight, scales: embedScales, biases: embedBiases)
                } else {
                    fullWeight = embedWeight
                }

                // Look up the input embedding row per token and mean-pool.
                let idxArr = MLXArray(tokenIds.map { Int32($0) })
                let rows = fullWeight.asType(.float32).take(idxArr, axis: 0)
                let pooled = rows.mean(axis: 0).reshaped(-1)

                let norm = MLX.sqrt((pooled * pooled).sum())
                let normalized = norm.item(Float.self) > 0 ? pooled / norm : pooled
                MLX.eval(normalized)

                let floats = normalized.asArray(Float.self)
                let buf = UnsafeMutablePointer<Float>.allocate(capacity: floats.count)
                buf.initialize(from: floats, count: floats.count)
                box.value = (buf, Int32(floats.count), Int32(tokenIds.count))
            }
        } catch {
            // box stays nil → returns -1
        }
        sema.signal()
    }
    sema.wait()

    guard let (buf, len, promptTokens) = box.value else { return -1 }
    outData.pointee = buf
    outLen.pointee = len
    return promptTokens
}

// ---------------------------------------------------------------------------
// mlx_free_floats
// ---------------------------------------------------------------------------

@_cdecl("mlx_free_floats")
public func mlxFreeFloats(_ data: UnsafeMutablePointer<Float>?) {
    data?.deallocate()
}
