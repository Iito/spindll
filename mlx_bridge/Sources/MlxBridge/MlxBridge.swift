import Foundation
import CryptoKit
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXLLM
import MLXVLM
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
// Each stored state is the live cache as generation left it, so it covers the
// prompt plus the tokens generated that turn — that is exactly the prefix the
// next turn's prompt extends.
//
// Entries keep that state as computed, in f16, so a restored prefix gives the
// same result as prefilling it again: output with the cache matches output
// without it, as in mlx-lm. The byte budget evicts least-recently-used entries
// (long ones spill to the disk tier first). Trading that exactness for memory
// is opt-in: under a KV scheme (`--kv-scheme`) entries stay in the scheme's
// compressed layout.
//
// Access is serialised by ModelContainer's SerialAccessContainer actor, so no
// additional lock is needed.
// ---------------------------------------------------------------------------

private let kvGroupSize = 64

/// Most prompt tokens per prefill step. mlx-swift-lm defaults to 512 and clears
/// MLX's buffer cache after every step; Python mlx-lm's server takes 2048 at a
/// time. Bigger steps spread each pass over the weights, and each cache clear,
/// across more tokens, for a higher activation peak while a long prompt fills.
/// A model with a sliding window still caps the step at its window.
private let prefillStepCeiling = 2048

/// `params` with the prefill step for handing `count` tokens to a TokenIterator.
///
/// mlx-swift-lm's generic prefill passes a prompt that fits in one step to the
/// iterator's first forward whole, and that forward projects every position
/// onto the vocabulary to sample from the last one: on Llama 3.2 1B, ~2k tokens
/// make about 1 TFLOP of work and 0.5 GB of logits thrown away. A longer prompt
/// is chunked, leaving only its last token for that forward. So a prompt past
/// the default step gets a step just under its length (at most the ceiling):
/// chunked, with one token left over, as in mlx-lm. A shorter prompt keeps the
/// single forward, cheaper there than another pass over the weights.
private func withPrefillStep(_ params: GenerateParameters, count: Int) -> GenerateParameters {
    var p = params
    p.prefill.stepSize = count > PrefillParameters.defaultStepSize
        ? min(prefillStepCeiling, count - 1) : nil
    return p
}

private let fullBits = 16    // `bits` of an entry kept as computed, in f16
private let highBits = 8     // width a KV scheme's leftover f16 layers are stored at

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
    /// Width of the attention layers: `fullBits` for an exact f16 entry, else
    /// the quantized width (a KV scheme's entry, or an older one from disk).
    var bits: Int
    /// In-RAM footprint of `kvCache`.
    var sizeBytes: Int64
    /// Model state that belongs with `kvCache`, as TokenIterator left it.
    /// Qwen-VL-family models need it to continue from a warm cache: their
    /// M-RoPE rope delta, stored relative to the cache offset, so it stays valid
    /// when the restored prefix is cut back. nil for most models.
    var state: LMOutput.State?
    /// A copy of this state has already been written to the on-disk tier — so
    /// it isn't re-written when the entry is later evicted.
    var spilled = false

    init(tokenIds: [Int32], kvCache: [any KVCache], bits: Int, state: LMOutput.State?) {
        self.tokenIds  = tokenIds
        self.kvCache   = kvCache
        self.bits      = bits
        self.state     = state
        self.sizeBytes = sizeOf(kvCache)
    }
}

/// Result of a prompt-cache lookup: the matched entry and how many leading
/// tokens it shares with the requested prompt.
private struct PromptCacheHit {
    let entry: PromptCacheEntry
    let prefixLength: Int
}

/// Bytes `kvCache` keeps in memory. A layer's `state` is a view of the tokens
/// it holds, while its buffers grow in steps (256 tokens for KVCacheSimple), so
/// each layer counts the buffers it holds (`innerState`), or the view for a
/// type that reports none.
private func sizeOf(_ kvCache: [any KVCache]) -> Int64 {
    var total: Int64 = 0
    for layer in kvCache {
        let used = layer.state.reduce(Int64(0)) { $0 + Int64($1.nbytes) }
        let held = ((layer as? BaseKVCache)?.innerState() ?? [])
            .reduce(Int64(0)) { $0 + Int64($1.nbytes) }
        total += max(used, held)
    }
    return total
}

/// `cache` with each plain attention layer whose buffer is more than a quarter
/// empty copied down to the tokens it holds. A short conversation otherwise
/// keeps most of a 256-token growth step per layer, which the byte budget now
/// counts. Other layer types are kept as they are.
private func trimmedToFit(_ cache: [any KVCache]) -> [any KVCache] {
    var pending: [MLXArray] = []
    let fitted: [any KVCache] = cache.map { layer in
        guard let simple = layer as? KVCacheSimple, type(of: simple) == KVCacheSimple.self
        else { return layer }
        let used = simple.state.reduce(0) { $0 + $1.nbytes }
        let held = simple.innerState().reduce(0) { $0 + $1.nbytes }
        guard (held - used) * 4 > held else { return layer }
        let copy = KVCacheSimple()
        copy.state = simple.state.map { $0 * 1 }   // `* 1` detaches; offset unchanged
        pending += copy.state
        return copy
    }
    if !pending.isEmpty { MLX.eval(pending) }
    return fitted
}

// ---------------------------------------------------------------------------
// Cache budgets — set once at startup via `mlx_set_cache_budgets`. Defaults
// match the GGUF side's flag defaults so behaviour is consistent if the Rust
// side never calls the setter.
// ---------------------------------------------------------------------------

private var mlxRamCacheBudget: Int64  = 512 * 1024 * 1024            // 512 MB
private var mlxDiskCacheBudget: Int64 = 2 * 1024 * 1024 * 1024       // 2 GB

/// Configure the prompt cache's RAM and disk byte budgets. Should be called
/// before any models are loaded — `sharedMlxDiskCache` reads `diskBytes` on
/// first access, and each `PromptCache` reads `ramBytes` at construction. Pass
/// 0 for either tier to effectively disable it.
@_cdecl("mlx_set_cache_budgets")
public func mlxSetCacheBudgets(_ ramBytes: UInt64, _ diskBytes: UInt64) {
    let clamp: (UInt64) -> Int64 = { Int64(min($0, UInt64(Int64.max))) }
    mlxRamCacheBudget  = clamp(ramBytes)
    mlxDiskCacheBudget = clamp(diskBytes)
    fputs("[mlx-bridge] prompt cache: ram=\(ramBytes >> 20) MB disk=\(diskBytes >> 20) MB\n", stderr)
}

// ---------------------------------------------------------------------------
// KV cache scheme — set once at startup via `mlx_set_kv_scheme`. nil keeps
// the default f16 KV cache. Otherwise it names one of mlx-swift-lm's
// compressed layouts, e.g. "turbo8v3" (TurboQuant: 8-bit keys, 3-bit values),
// "turbo0v4" or "affine8"; GenerateParameters.kvScheme documents the set.
// ---------------------------------------------------------------------------

private var mlxKVScheme: String? = nil

/// Pick the KV cache scheme for every later generation, or clear it with
/// NULL. Returns 0 on success, or -1 with the setting unchanged when
/// mlx-swift-lm doesn't know the name. Call before any model is loaded:
/// each model's prompt-cache namespace includes the scheme, fixed at load.
@_cdecl("mlx_set_kv_scheme")
public func mlxSetKVScheme(_ scheme: UnsafePointer<CChar>?) -> Int32 {
    guard let scheme else {
        mlxKVScheme = nil
        return 0
    }
    let name = String(cString: scheme)
    guard resolveAffineScheme(name) != nil || resolveTurboScheme(name) != nil
            || resolveVarianceNormalizedScheme(name) != nil else {
        fputs("[mlx-bridge] unknown KV cache scheme \"\(name)\"\n", stderr)
        return -1
    }
    mlxKVScheme = name
    fputs("[mlx-bridge] KV cache scheme: \(name)\n", stderr)
    return 0
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

    init() {
        self.maxBytes = mlxDiskCacheBudget
        self.enabled = mlxDiskCacheBudget > 0
            && ProcessInfo.processInfo.environment["SPINDLL_MLX_DISK_CACHE"] != "0"
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
        -> (tokenIds: [Int32], kvCache: [any KVCache], bits: Int, state: LMOutput.State?)? {
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

        let snapshot: PromptCacheSnapshot
        do { snapshot = try loadPromptCacheSnapshot(url: url) }
        catch {
            fputs("[mlx-bridge] disk cache: load failed (\(entry.file)): \(error)\n", stderr)
            lock.lock(); index[entry.file] = nil; saveIndexLocked(); lock.unlock()
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        // loadPromptCache rebuilds QuantizedKVCache layers at the default 8-bit
        // width — mlx-swift-lm doesn't restore bits/groupSize from metaState — so
        // re-bind the loaded arrays into caches with the width we recorded.
        let loaded = snapshot.cache
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
        return (entry.tokenIds, kvCache, entry.bits, snapshot.state)
    }

    /// Write an evicted RAM entry to disk, with the model state that belongs
    /// with it, dropping LRU disk entries if over budget.
    func put(modelDigest: String, tokenIds: [Int32], kvCache: [any KVCache], bits: Int,
             state: LMOutput.State?) {
        guard enabled, !tokenIds.isEmpty else { return }
        var tokenData = Data(capacity: tokenIds.count * 4)
        for t in tokenIds { withUnsafeBytes(of: t.littleEndian) { tokenData.append(contentsOf: $0) } }
        let file = "\(modelDigest)_\(sha256Hex(tokenData)).safetensors"
        let url = dir.appendingPathComponent(file)
        do { try savePromptCache(url: url, cache: kvCache, metadata: [:], state: state) }
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
    /// Total in-RAM byte budget for the cache. Configured via `mlx_set_cache_budgets`
    /// (wired to `--kv-ram-cache`); `0` disables the prompt cache entirely.
    private let maxBytes: Int64
    /// Shortest shared prefix worth restoring. A restore copies only the shared
    /// tokens, in one GPU pass, while each prompt token it skips saves a whole
    /// forward pass of prefill, so even the chat-template header every prompt
    /// opens with (system prompt and role markers, often 20-40 tokens) pays off,
    /// most of all on large models. Shorter shares save too little to matter.
    private let minPrefix: Int
    /// Only touch the on-disk tier for prompts at least this long — for short
    /// prompts, reading an ~tens-of-MB safetensors back costs more than just
    /// re-prefilling. The RAM tier is used regardless of length.
    private let diskMinTokens = 512
    /// How many more prompt tokens than the RAM hit a disk entry must cover
    /// before loading it is worth the I/O.
    private let diskMinGain = 64
    /// Model identity (digest of config.json) — namespaces this model's entries
    /// in the shared on-disk tier.
    private let modelDigest: String

    init(modelDigest: String, maxBytes: Int64? = nil, minPrefix: Int = 16) {
        self.modelDigest = modelDigest
        self.maxBytes    = maxBytes ?? mlxRamCacheBudget
        self.minPrefix   = minPrefix
    }

    private var usedBytes: Int64 { entries.reduce(0) { $0 + $1.sizeBytes } }

    /// RAM lookup: the entry sharing the longest token prefix with `tokenIds`
    /// (≥ `minPrefix`), promoted to the front of the LRU list. Among entries
    /// that share equally far (every conversation shares the chat header), the
    /// finer snapshot wins, then the most recent.
    private func lookupRam(_ tokenIds: [Int32]) -> PromptCacheHit? {
        var bestIdx: Int? = nil
        var bestLen = 0
        for (i, e) in entries.enumerated() {
            let l = commonPrefixLength(e.tokenIds, tokenIds)
            let better = l > bestLen
                || (l == bestLen && bestIdx.map { e.bits > entries[$0].bits } == true)
            if better { bestLen = l; bestIdx = i }
        }
        guard let idx = bestIdx, bestLen >= minPrefix else { return nil }
        let entry = entries.remove(at: idx)
        entries.insert(entry, at: 0)
        return PromptCacheHit(entry: entry, prefixLength: bestLen)
    }

    /// Look up `tokenIds`, checking RAM first and then the on-disk tier; a disk
    /// hit is loaded back into RAM (when RAM tier is enabled). `nil` if neither
    /// has a usable prefix. `maxBytes == 0` disables only the RAM tier — disk
    /// lookups still run so callers using `--no-kv-ram-cache` keep their disk
    /// cache.
    func lookup(tokenIds: [Int32]) -> PromptCacheHit? {
        guard !tokenIds.isEmpty else { return nil }
        let ram = maxBytes > 0 ? lookupRam(tokenIds) : nil
        // A RAM hit can cover as little as the chat header, so a long prompt
        // still asks the disk tier, which may hold far more of it.
        let covered = ram?.prefixLength ?? 0
        guard tokenIds.count >= diskMinTokens,
              tokenIds.count - covered >= diskMinGain,
              let d = sharedMlxDiskCache.lookup(
                modelDigest: modelDigest, tokenIds: tokenIds,
                minPrefix: max(minPrefix, covered + diskMinGain))
        else { return ram }
        let entry = PromptCacheEntry(tokenIds: d.tokenIds, kvCache: d.kvCache, bits: d.bits,
                                     state: d.state)
        entry.spilled = true   // it came from disk; no need to write it back
        if maxBytes > 0 {
            entries.removeAll { commonPrefixLength($0.tokenIds, entry.tokenIds) == $0.tokenIds.count }
            entries.insert(entry, at: 0)
            evictExcess()
        }
        return PromptCacheHit(entry: entry, prefixLength: commonPrefixLength(entry.tokenIds, tokenIds))
    }

    /// Insert (or replace) the entry for `tokenIds`. Any existing entry whose
    /// token sequence is a prefix of the new one is dropped (the new entry
    /// supersedes it). Surviving older entries spill a copy to the on-disk tier,
    /// once. While the cache is over its byte budget, the LRU entry is dropped
    /// from RAM (already on disk).
    func save(tokenIds: [Int32], kvCache: [any KVCache], bits: Int, state: LMOutput.State?) {
        guard !tokenIds.isEmpty else { return }
        let entry = PromptCacheEntry(tokenIds: tokenIds, kvCache: kvCache, bits: bits, state: state)
        // RAM tier disabled: skip RAM bookkeeping but still spill to disk so
        // `--no-kv-ram-cache` users keep their on-disk cache.
        if maxBytes == 0 {
            spillToDisk(entry)
            return
        }
        entries.removeAll { commonPrefixLength($0.tokenIds, tokenIds) == $0.tokenIds.count }
        for e in entries { spillToDisk(e) }
        entries.insert(entry, at: 0)
        evictExcess()
    }

    /// Take `entry` out of the RAM tier because its state is becoming a live
    /// cache: generation extends it, so its token list and size would no longer
    /// describe it. The save after generation stores the extended state, and a
    /// turn that never gets there leaves no stale entry behind.
    func take(_ entry: PromptCacheEntry) {
        entries.removeAll { $0 === entry }
    }

    private func evictExcess() {
        while usedBytes > maxBytes, let evicted = entries.popLast() {
            spillToDisk(evicted)   // no-op if already spilled
        }
    }

    /// Write `entry`'s state to the on-disk tier once, if it's long enough to
    /// be worth the I/O.
    private func spillToDisk(_ entry: PromptCacheEntry) {
        guard !entry.spilled, entry.tokenIds.count >= diskMinTokens else { return }
        sharedMlxDiskCache.put(modelDigest: modelDigest,
                               tokenIds: entry.tokenIds, kvCache: entry.kvCache, bits: entry.bits,
                               state: entry.state)
        entry.spilled = true
    }
}

// Retains the ModelContainer and prompt KV cache across FFI calls.
private final class ModelState: @unchecked Sendable {
    let container: ModelContainer
    let promptCache: PromptCache
    /// True when the model was loaded via VLMModelFactory (supports images).
    let isVLM: Bool
    /// False when the tokenizer cannot render its own chat template — text
    /// requests then fall back to ChatML and vision requests are rejected.
    let hasChatTemplate: Bool
    init(_ container: ModelContainer, modelDigest: String, isVLM: Bool = false,
         hasChatTemplate: Bool = true) {
        self.container = container
        self.promptCache = PromptCache(modelDigest: modelDigest)
        self.isVLM = isVLM
        self.hasChatTemplate = hasChatTemplate
    }
}

/// Prompt tokens as TokenIterator input. A model loaded as a VLM gets them as
/// [1, L], the shape its own processors build: Qwen3-VL's single-step prefill
/// reads `dim(1)` of the ids, so a flat array fails there. The generic LLM
/// prefill takes the flat [L] array.
private func promptTokenArray(_ tokens: [Int32], vlm: Bool) -> MLXArray {
    let array = MLXArray(tokens)
    return vlm ? array.expandedDimensions(axis: 0) : array
}

/// True when the tokenizer renders its own chat template. Any render failure
/// counts as "no template" — generation would take the ChatML fallback either
/// way.
private func probeChatTemplate(_ container: ModelContainer) async -> Bool {
    (try? await container.perform(nonSendable: ()) { context, _ in
        _ = try context.tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": "probe"]], tools: nil)
    }) != nil
}

/// One loud, load-time warning when a model ships no usable chat template —
/// per #82 the silent per-request fallback hid image-blind VLM generations.
private func warnMissingChatTemplate(modelPath: String, isVLM: Bool) {
    fputs("[mlx-bridge] WARNING: no chat template in \(modelPath)\n", stderr)
    if isVLM {
        fputs("[mlx-bridge]   This model is multimodal: image requests will be REJECTED, because\n", stderr)
        fputs("[mlx-bridge]   the template is what inserts the image tokens vision needs.\n", stderr)
    }
    fputs("[mlx-bridge]   Text requests fall back to generic ChatML (wrong special tokens for\n", stderr)
    fputs("[mlx-bridge]   most models). Newer HF repos ship the template as chat_template.jinja —\n", stderr)
    fputs("[mlx-bridge]   re-pull the model to fetch it.\n", stderr)
}

// ---------------------------------------------------------------------------
// mlx_model_load
// ---------------------------------------------------------------------------

/// Load a model from a local MLX-format directory.
/// Blocks the calling thread until the model is fully resident in memory.
/// Tries VLM first (supports vision models), falls back to LLM.
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
    // Cached KV state is in the layout of the scheme that wrote it, so the
    // namespace carries the scheme too: a restart under another scheme must
    // not pick up on-disk entries in a layout it doesn't use.
    let cacheNamespace = mlxKVScheme.map { "\(modelDigest)-\($0)" } ?? modelDigest

    Task {
        // Try VLM first — covers vision-language models (Qwen2.5-VL, Gemma3, etc.)
        // plus text-only models registered in VLMTypeRegistry (Qwen3.5, etc.).
        do {
            let container = try await VLMModelFactory.shared.loadContainer(
                from: url,
                using: #huggingFaceTokenizerLoader()
            )
            let hasTemplate = await probeChatTemplate(container)
            if !hasTemplate { warnMissingChatTemplate(modelPath: modelPath, isVLM: true) }
            box.value = ModelState(container, modelDigest: cacheNamespace, isVLM: true,
                                   hasChatTemplate: hasTemplate)
        } catch {
            // Fall back to LLM for text-only architectures.
            do {
                let container = try await LLMModelFactory.shared.loadContainer(
                    from: url,
                    using: #huggingFaceTokenizerLoader()
                )
                let hasTemplate = await probeChatTemplate(container)
                if !hasTemplate { warnMissingChatTemplate(modelPath: modelPath, isVLM: false) }
                box.value = ModelState(container, modelDigest: cacheNamespace, isVLM: false,
                                       hasChatTemplate: hasTemplate)
            } catch {
                // box.value stays nil; Rust side receives NULL
            }
        }
        sema.signal()
    }
    sema.wait()

    guard let state = box.value else { return nil }
    return Unmanaged.passRetained(state).toOpaque()
}

/// Returns 1 if the model was loaded as a VLM (supports vision), 0 otherwise.
@_cdecl("mlx_model_is_vlm")
public func mlxModelIsVlm(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard let handle else { return 0 }
    let state = Unmanaged<ModelState>.fromOpaque(handle).takeUnretainedValue()
    return state.isVLM ? 1 : 0
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
            // No prompt cache here, so TokenIterator can convert the cache to
            // the scheme itself.
            let params = GenerateParameters(
                maxTokens: Int(maxTokens),
                kvScheme: mlxKVScheme,
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
                let lmInput = LMInput(
                    tokens: promptTokenArray(tokenIds.map { Int32($0) }, vlm: state.isVLM))
                var iterator = try TokenIterator(
                    input: lmInput,
                    model: context.model,
                    parameters: withPrefillStep(params, count: tokenIds.count)
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

                // Flush the text the detokenizer is still holding back. It
                // withholds text that ends in an incomplete UTF-8 sequence
                // (maxTokens hit mid-codepoint, say), and calling next() again
                // would keep withholding it. finish() emits it, with U+FFFD
                // standing in for the incomplete bytes.
                if !cancelled, let chunk = detokenizer.finish() {
                    _ = chunk.withCString { ptr in callback(ptr, callbackCtx) }
                }

                // Wait for in-flight async evaluations to finish before the
                // perform closure tears down — mirrors upstream's sync loop
                // in `runSynchronousGenerationLoop`. It has to be the default
                // stream, which TokenIterator evaluates on: since mlx-swift
                // 0.33, `Stream()` makes a new, empty stream, and waiting on
                // that one returns at once.
                Stream.defaultStream.synchronize()
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
/// Recursively convert a `JSONSerialization` value into a `Sendable` value.
/// swift-transformers' `ToolSpec` is `[String: any Sendable]`, but
/// `JSONSerialization` yields non-`Sendable` nested `NSDictionary`/`NSArray`, so
/// rebuild the structure with Sendable containers.
private func jsonToSendable(_ value: Any) -> any Sendable {
    switch value {
    case let s as String: return s
    case let n as NSNumber: return n
    case let a as [Any]: return a.map(jsonToSendable)
    case let d as [String: Any]: return d.mapValues(jsonToSendable)
    default: return ""  // JSON null / unexpected → empty string (Sendable)
    }
}

/// Decode a JSON array of OpenAI tool specs (`[{type, function}]`) into the
/// `[[String: any Sendable]]` (== `[ToolSpec]`) shape that
/// `applyChatTemplate(messages:tools:)` expects. Returns nil for a
/// null/empty/invalid pointer so no `tools` variable is bound in the template.
private func decodeTools(_ toolsJson: UnsafePointer<CChar>?) -> [[String: any Sendable]]? {
    guard let toolsJson,
          let data = String(cString: toolsJson).data(using: .utf8),
          let parsed = try? JSONSerialization.jsonObject(with: data),
          let arr = parsed as? [Any]
    else { return nil }
    let tools = arr.compactMap { ($0 as? [String: Any])?.mapValues(jsonToSendable) }
    return tools.isEmpty ? nil : tools
}

private func applyChatTemplateWithFallback(
    tokenizer: any MLXLMCommon.Tokenizer,
    messages: [[String: String]],
    tools: [[String: any Sendable]]?
) throws -> [Int] {
    do {
        return try tokenizer.applyChatTemplate(messages: messages, tools: tools)
    } catch is MLXLMCommon.TokenizerError {
        fputs("[mlx-bridge] model has no chat template, using ChatML fallback " +
              "(re-pull to fetch chat_template.jinja if the repo ships one)\n", stderr)
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
// Restoring and storing prompt-cache state
// ---------------------------------------------------------------------------

/// An independent copy of `cache` cut back to its first `keep` tokens, for
/// restoring a cached prefix. Plain attention layers copy only those tokens,
/// so reusing just the chat header of a long entry doesn't drag the rest of
/// that entry along; with `toF16`, quantized ones come back as f16 (see the
/// hit path for why). Other layer types (a scheme's layout, a sliding-window
/// ring) are copied whole and rewound. Every copy is evaluated in one GPU
/// pass. `nil` when some layer can't rewind that far, such as a ring already
/// past its window: the caller then prefills from scratch.
private func restorePrefix(_ cache: [any KVCache], keep: Int, toF16: Bool) -> [any KVCache]? {
    guard keep > 0, cache.allSatisfy({ $0.offset >= keep }) else { return nil }
    let cut = { (a: MLXArray) -> MLXArray in a[.ellipsis, ..<keep, 0...] }
    var restored: [any KVCache] = []
    var rewind: [any KVCache] = []
    var pending: [MLXArray] = []
    for layer in cache {
        // Exact classes only: a subclass may hold arrays that aren't laid out
        // by token, which a cut along the token axis would corrupt.
        if let simple = layer as? KVCacheSimple, type(of: simple) == KVCacheSimple.self {
            let copy = KVCacheSimple()
            copy.state = simple.state.map { cut($0) * 1 }   // `* 1` detaches; offset = keep
            restored.append(copy)
            pending += copy.state
        } else if let q = layer as? QuantizedKVCache, type(of: q) == QuantizedKVCache.self,
                  let prefix = q.copy() as? QuantizedKVCache {
            // Views of the first `keep` tokens, with groupSize, bits and mode.
            prefix.state = q.state.map(cut)
            prefix.offset = keep
            let copy: any KVCache
            if toF16 {
                copy = prefix.toUnquantized()   // fresh arrays: dequantizes just `keep` tokens
            } else {
                prefix.state = prefix.state.map { $0 * 1 }
                copy = prefix
            }
            restored.append(copy)
            pending += copy.state
        } else {
            guard canTrimPromptCache([layer], numTokens: layer.offset - keep) else { return nil }
            // mlx-swift-lm's copy() returns views of the original's buffers,
            // which generation would then write through; detach them.
            var copy = layer.copy()
            copy.state = copy.state.map { $0 * 1 }
            restored.append(copy)
            rewind.append(copy)
            pending += copy.state
        }
    }
    MLX.eval(pending)
    for layer in rewind {
        let n = layer.offset - keep
        guard layer.trim(n) == n else { return nil }
    }
    return restored
}

/// Store form of a KV scheme's live cache whose generation has finished.
/// Layers already compressed (the scheme's own, or the 8-bit boundary layers
/// it protects) are kept as they are: nothing mutates them any more, so they
/// need no copy. Only f16 layers the scheme left alone (a head size it can't
/// group, a one-token prompt) are quantized to `bits`.
private func compactFinishedCache(_ cache: [any KVCache], bits: Int) throws -> [any KVCache] {
    try cache.map { layer in
        guard let simple = layer as? KVCacheSimple else { return layer }
        return try simple.toQuantized(groupSize: kvGroupSize, bits: bits)
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
    _ toolsJson:   UnsafePointer<CChar>?,
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
    let tools = decodeTools(toolsJson)

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
                    tokenizer: context.tokenizer, messages: msgs, tools: tools)
                let tokenIds = rawIds.map { Int32($0) }

                // --- Prompt KV cache (prefix reuse) ---
                // Look for a cached state sharing a token prefix with this
                // prompt. On a hit we restore that state, trim it back to the
                // shared-prefix length, and prefill only the new tokens; on a
                // miss we prefill the whole prompt into a fresh cache. Either
                // way the live cache, now covering prompt + generated tokens,
                // is stored after generation for the next prefix lookup.
                //
                // `canTrimPromptCache` is the SWA fallback: for Gemma-style
                // sliding-window models `RotatingKVCache.isTrimmable` flips to
                // false once the conversation exceeds the window, in which case
                // we silently fall back to a cold prefill, as we do when a
                // layer can't rewind as far as the restore needs. Same guard on
                // the save path below skips storing non-trimmable caches.
                let liveCache: [any KVCache]
                var iterator: TokenIterator
                let storedBits: Int     // `bits` the live cache is stored with
                let compactOnSave: Bool // a scheme miss: compact leftover f16 layers

                // A hit restores the shared prefix, minus the last prompt token
                // when the whole prompt is cached: TokenIterator needs at least
                // one token to produce the logits it samples from.
                //
                // Without a KV scheme entries are exact f16, so the restored
                // prefix computes what a cold prefill would and the output
                // matches. An entry that is exactly a prefix of this prompt is
                // consumed in place: it leaves the cache, and the save below
                // stores it extended. A quantized entry, such as an 8-bit one
                // read back from disk, is dequantized into a copy instead of
                // extended: appending to it would leave the whole new
                // conversation at its width. Under a scheme the restored state
                // keeps the scheme's layout. Either way the entry's model state
                // comes along, which Qwen-VL-family models need on a warm cache.
                var restoredHit: (cache: [any KVCache], keep: Int, bits: Int,
                                  modelState: LMOutput.State?)? = nil
                if let hit = state.promptCache.lookup(tokenIds: tokenIds),
                   canTrimPromptCache(hit.entry.kvCache) {
                    let cachedLen = hit.entry.tokenIds.count
                    let keep = min(hit.prefixLength, tokenIds.count - 1)
                    let extendable = mlxKVScheme != nil || hit.entry.bits == fullBits
                    if extendable && hit.prefixLength == cachedLen {
                        let n = cachedLen - keep
                        if n == 0 || canTrimPromptCache(hit.entry.kvCache, numTokens: n) {
                            // Its layers become the live cache, so the entry
                            // leaves the RAM tier before anything changes them.
                            state.promptCache.take(hit.entry)
                            if n == 0 || trimPromptCache(hit.entry.kvCache, numTokens: n) == n {
                                restoredHit = (hit.entry.kvCache, keep, hit.entry.bits,
                                               hit.entry.state)
                            }
                        }
                    } else if let copy = restorePrefix(hit.entry.kvCache, keep: keep,
                                                       toF16: mlxKVScheme == nil) {
                        restoredHit = (copy, keep, mlxKVScheme == nil ? fullBits : hit.entry.bits,
                                       hit.entry.state)
                    }
                }

                if let hit = restoredHit {
                    liveCache = hit.cache
                    storedBits = hit.bits
                    compactOnSave = false
                    iterator = try TokenIterator(
                        input: LMInput(tokens: promptTokenArray(Array(tokenIds[hit.keep...]),
                                                                vlm: state.isVLM)),
                        model: context.model,
                        cache: liveCache,
                        state: hit.modelState,
                        parameters: withPrefillStep(params, count: tokenIds.count - hit.keep)
                    )
                } else {
                    var fresh = try makePromptCache(model: context.model, parameters: params)
                    var seedTokens = tokenIds
                    var seedState: LMOutput.State? = nil
                    if let scheme = mlxKVScheme, tokenIds.count > 1 {
                        // Compress once the prompt is in, as upstream does, but
                        // on our own array. Handed the scheme, TokenIterator
                        // would swap in new layer objects that `liveCache`
                        // never sees, and the save below would store the stale
                        // f16 layers. So prefill all but the last prompt token
                        // here, convert, and let the iterator take the last one.
                        // `params` carries no scheme, so it never converts again.
                        let warm = try TokenIterator(
                            input: LMInput(tokens: promptTokenArray(Array(tokenIds.dropLast()),
                                                                    vlm: state.isVLM)),
                            model: context.model,
                            cache: fresh,
                            parameters: withPrefillStep(params, count: tokenIds.count - 1)
                        )
                        seedState = warm.state
                        maybeQuantizeKVCache(cache: &fresh, kvBits: nil, kvScheme: scheme)
                        seedTokens = [tokenIds[tokenIds.count - 1]]
                    }
                    liveCache = fresh
                    storedBits = mlxKVScheme == nil ? fullBits : highBits
                    compactOnSave = mlxKVScheme != nil
                    iterator = try TokenIterator(
                        input: LMInput(tokens: promptTokenArray(seedTokens, vlm: state.isVLM)),
                        model: context.model,
                        cache: liveCache,
                        state: seedState,
                        parameters: withPrefillStep(params, count: seedTokens.count)
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

                if !cancelled, let chunk = detokenizer.finish() {
                    _ = chunk.withCString { ptr in callback(ptr, callbackCtx) }
                }

                // Hand the live cache — now covering prompt + generated tokens —
                // to the prompt cache for the next turn's prefix lookup. Skip
                // non-trimmable caches (Mamba/SSM hybrids): they could only ever
                // be matched whole, which is never useful for prefix reuse.
                //
                // Generation is over and nothing mutates the live cache any more,
                // so it is stored as it is: exact f16 by default, the scheme's
                // layout under a KV scheme. It shares no buffers with any other
                // entry: a hit restores into a copy, and an entry consumed in
                // place has already left the cache. Only a layer whose buffer is
                // mostly unused is copied down to size (`trimmedToFit`), since
                // the byte budget counts whole buffers. A scheme miss can still
                // hold f16 layers the scheme left alone, which
                // `compactFinishedCache` quantizes. The iterator's model state
                // is stored with the cache, for the next turn to continue from.
                if canTrimPromptCache(liveCache) {
                    // A failed compaction only loses the cache entry — the
                    // generation already streamed, so never fail it here.
                    let toStore = compactOnSave
                        ? (try? compactFinishedCache(liveCache, bits: storedBits))
                        : trimmedToFit(liveCache)
                    if let toStore {
                        state.promptCache.save(
                            tokenIds: tokenIds + generatedTokenIds,
                            kvCache: toStore,
                            bits: storedBits,
                            state: iterator.state)
                    }
                }

                Stream.defaultStream.synchronize()
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
// Image resizing helper
// ---------------------------------------------------------------------------

/// Maximum pixel count before an input image is pre-scaled. Default ~1M px
/// (≈ 1024×1024): Qwen-VL vision prefill scales linearly with patch count
/// while decode speed is unaffected, so pixels beyond this mostly buy prefill
/// latency — models ship permissive caps (qwen3.5-9b: 16.8M px). Override
/// with SPINDLL_VLM_MAX_PIXELS; 0 or negative disables the cap.
private let kDefaultMaxPixels: Int = {
    if let raw = ProcessInfo.processInfo.environment["SPINDLL_VLM_MAX_PIXELS"],
       let v = Int(raw) {
        return v <= 0 ? Int.max : v
    }
    return 1_048_576
}()

/// If the image at `url` exceeds `maxPixels` total pixels, write a
/// proportionally down-scaled JPEG (preserving colour and aspect ratio)
/// to a temporary file and return its URL. Otherwise returns the
/// original URL unchanged.
private func resizeImageIfNeeded(url: URL, maxPixels: Int = kDefaultMaxPixels) -> URL {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else { return url }

    let w = cgImage.width
    let h = cgImage.height
    let totalPixels = w * h

    guard totalPixels > maxPixels else { return url }

    // Scale factor to fit within maxPixels while preserving aspect ratio.
    let scale = sqrt(Double(maxPixels) / Double(totalPixels))
    let newW = Int(Double(w) * scale)
    let newH = Int(Double(h) * scale)

    // Draw into an sRGB bitmap context — preserves colour.
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: nil,
        width: newW,
        height: newH,
        bitsPerComponent: 8,
        bytesPerRow: newW * 4,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return url }

    ctx.interpolationQuality = .high
    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: newW, height: newH))

    guard let resized = ctx.makeImage() else { return url }

    // Write as JPEG to a temp file.
    let tmpPath = NSTemporaryDirectory() + "spindll_vlm_resized_\(UUID().uuidString).jpg"
    let tmpURL = URL(fileURLWithPath: tmpPath)
    guard let dest = CGImageDestinationCreateWithURL(
        tmpURL as CFURL, UTType.jpeg.identifier as CFString, 1, nil
    ) else { return url }

    // Quality 0.85 — good fidelity, reasonable file size.
    let opts: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: 0.85]
    CGImageDestinationAddImage(dest, resized, opts as CFDictionary)
    guard CGImageDestinationFinalize(dest) else { return url }

    fputs("[mlx-bridge] resized image from \(w)×\(h) to \(newW)×\(newH) (\(tmpPath))\n", stderr)
    return tmpURL
}

// ---------------------------------------------------------------------------
// mlx_chat_generate_vision
// ---------------------------------------------------------------------------

/// Chat generation with image input for VLM models.
///
/// `messagesJson` is a UTF-8 JSON array of `{"role": ..., "content": ...}`.
/// `imagePath` is a NUL-terminated path to an image file on disk (JPEG, PNG, etc.),
/// or NULL for text-only requests. The image is attached to the last user message.
///
/// Returns the number of tokens generated on success, -1 on error.
@_cdecl("mlx_chat_generate_vision")
public func mlxChatGenerateVision(
    _ handle:      UnsafeMutableRawPointer?,
    _ messagesJson: UnsafePointer<CChar>?,
    _ imagePath:   UnsafePointer<CChar>?,
    _ maxTokens:   UInt32,
    _ temperature: Float,
    _ topP:        Float,
    _ seed:        UInt32,
    _ callback:    @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Int32,
    _ callbackCtx: UnsafeMutableRawPointer?
) -> Int32 {
    guard let handle, let messagesJson else { return -1 }
    let state = Unmanaged<ModelState>.fromOpaque(handle).takeUnretainedValue()
    // Without the model's real template the image tokens vision conditioning
    // needs are never inserted — a ChatML-formatted request would generate
    // image-blind. Fail fast instead (#82).
    guard state.hasChatTemplate else {
        fputs("[mlx-bridge] vision request rejected: model has no chat template " +
              "(re-pull to fetch chat_template.jinja)\n", stderr)
        return -1
    }
    let json = String(cString: messagesJson)

    guard
        let data = json.data(using: .utf8),
        let parsed = try? JSONSerialization.jsonObject(with: data),
        let messages = parsed as? [[String: String]]
    else { return -1 }

    // Resolve image from file path if provided, resizing if too large.
    var userImage: UserInput.Image? = nil
    var resizedTempURL: URL? = nil
    if let imagePath {
        let pathStr = String(cString: imagePath)
        let url = URL(fileURLWithPath: pathStr)
        let attrs = try? FileManager.default.attributesOfItem(atPath: pathStr)
        let fileSize = (attrs?[.size] as? UInt64) ?? 0
        fputs("[mlx-bridge] image path: \(pathStr) (\(fileSize) bytes)\n", stderr)
        guard fileSize > 0 else {
            fputs("[mlx-bridge] image file not found or empty: \(pathStr)\n", stderr)
            return -1
        }
        let finalURL = resizeImageIfNeeded(url: url)
        if finalURL != url { resizedTempURL = finalURL }
        userImage = .url(finalURL)
    }

    let sema = DispatchSemaphore(value: 0)
    let box = Box<Int32>()

    Task {
        do {
            // No prompt cache on the vision path, so TokenIterator can convert
            // the cache to the scheme itself. Vision models prefill through
            // their own paths, so they keep the library's prefill step.
            let params = GenerateParameters(
                maxTokens: Int(maxTokens),
                kvScheme: mlxKVScheme,
                temperature: temperature,
                topP: topP
            )

            // Build UserInput from chat messages + optional image.
            // Find the index of the last user message so we can attach the image to it.
            var lastUserIdx: Int? = nil
            for (i, msg) in messages.enumerated() {
                if msg["role"] == "user" { lastUserIdx = i }
            }

            var chatMessages: [Chat.Message] = []
            for (i, msg) in messages.enumerated() {
                guard let role = msg["role"], let content = msg["content"] else { continue }
                switch role {
                case "system":
                    chatMessages.append(.system(content))
                case "assistant":
                    chatMessages.append(.assistant(content))
                default:
                    if let img = userImage, i == lastUserIdx {
                        chatMessages.append(.user(content, images: [img]))
                    } else {
                        chatMessages.append(.user(content))
                    }
                }
            }

            let userInput = UserInput(chat: chatMessages)

            var generated: Int32 = 0
            fputs("[mlx-bridge] userInput has \(userInput.images.count) image(s)\n", stderr)
            try await state.container.perform(nonSendable: userInput) { context, input in
                fputs("[mlx-bridge] preparing input with processor: \(type(of: context.processor))\n", stderr)
                let lmInput = try await context.processor.prepare(input: input)
                let hasImage = lmInput.image != nil
                if let img = lmInput.image {
                    let shape = img.pixels.shape
                    let mean = img.pixels.mean().item(Float.self)
                    let minVal = img.pixels.min().item(Float.self)
                    let maxVal = img.pixels.max().item(Float.self)
                    fputs("[mlx-bridge] lmInput: tokens=\(lmInput.text.tokens.count) image shape=\(shape) mean=\(mean) min=\(minVal) max=\(maxVal)\n", stderr)
                } else {
                    fputs("[mlx-bridge] lmInput: tokens=\(lmInput.text.tokens.count) hasImage=false\n", stderr)
                }
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
                        if shouldContinue == 0 {
                            cancelled = true
                            break
                        }
                    }
                }

                if !cancelled, let chunk = detokenizer.finish() {
                    _ = chunk.withCString { ptr in callback(ptr, callbackCtx) }
                }

                Stream.defaultStream.synchronize()
            }
            box.value = generated
        } catch {
            fputs("[mlx-bridge] mlx_chat_generate_vision error: \(error)\n", stderr)
            box.value = -1
        }
        sema.signal()
    }
    sema.wait()

    // Clean up resized temp file if we created one.
    if let tmp = resizedTempURL {
        try? FileManager.default.removeItem(at: tmp)
    }

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
    _ messagesJson: UnsafePointer<CChar>?,
    _ toolsJson: UnsafePointer<CChar>?
) -> UnsafePointer<CChar>? {
    guard let handle, let messagesJson else { return nil }
    let state = Unmanaged<ModelState>.fromOpaque(handle).takeUnretainedValue()
    let json = String(cString: messagesJson)
    let tools = decodeTools(toolsJson)

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
                    tokenizer: context.tokenizer, messages: messages, tools: tools)
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
