import Foundation
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
// On a cache hit for an
// identical prompt the expensive prefill is skipped: only the last prompt token
// is re-processed (one decode step), so TTFT ≈ single-token forward pass.
//
// Access is serialised by ModelContainer's SerialAccessContainer actor, so no
// additional lock is needed.
// ---------------------------------------------------------------------------

private final class PromptCacheEntry {
    let tokenIds: [Int32]
    let lastTokenId: Int32
    let kvCache: [any KVCache]

    init(tokenIds: [Int32], kvCache: [any KVCache]) {
        self.tokenIds  = tokenIds
        self.lastTokenId = tokenIds.last ?? 0
        self.kvCache   = kvCache
    }
}

private final class PromptCache {
    private var entries: [PromptCacheEntry] = []
    private let maxSize: Int

    init(maxSize: Int = 2) { self.maxSize = maxSize }

    /// Return the entry whose token sequence exactly matches, and promote it
    /// to the front of the LRU list.
    func lookup(tokenIds: [Int32]) -> PromptCacheEntry? {
        guard !tokenIds.isEmpty,
              let idx = entries.firstIndex(where: { $0.tokenIds == tokenIds })
        else { return nil }
        let entry = entries.remove(at: idx)
        entries.insert(entry, at: 0)
        return entry
    }

    /// Insert (or replace) an entry for the given token sequence, evicting the
    /// least-recently-used entry if the cache is full.
    func save(tokenIds: [Int32], kvCache: [any KVCache]) {
        guard !tokenIds.isEmpty else { return }
        entries.removeAll { $0.tokenIds == tokenIds }
        entries.insert(PromptCacheEntry(tokenIds: tokenIds, kvCache: kvCache), at: 0)
        if entries.count > maxSize { entries.removeLast() }
    }
}

// Retains the ModelContainer and prompt KV cache across FFI calls.
private final class ModelState: @unchecked Sendable {
    let container: ModelContainer
    let promptCache = PromptCache()
    init(_ container: ModelContainer) { self.container = container }
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

    Task {
        do {
            let container = try await LLMModelFactory.shared.loadContainer(
                from: url,
                using: #huggingFaceTokenizerLoader()
            )
            box.value = ModelState(container)
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
            box.value = -1
        }
        sema.signal()
    }
    sema.wait()

    return box.value ?? -1
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
                let rawIds  = try context.tokenizer.applyChatTemplate(messages: msgs)
                let tokenIds = rawIds.map { Int32($0) }

                // --- Prompt KV cache ---
                // On a hit we restore the saved KV state (trimmed by 1 token),
                // then feed only the last prompt token to TokenIterator so that
                // prefill is a single decode step instead of N steps.
                // On a miss we run full prefill and snapshot the resulting cache
                // (before generation) for the next call with the same prompt.
                var iterator: TokenIterator
                if let entry = state.promptCache.lookup(tokenIds: tokenIds) {
                    // HIT: restore cache at offset N-1, run last token only.
                    let restoredCache = entry.kvCache.map { $0.copy() }
                    trimPromptCache(restoredCache, numTokens: 1)
                    let seedInput = LMInput(tokens: MLXArray([entry.lastTokenId]))
                    iterator = try TokenIterator(
                        input: seedInput,
                        model: context.model,
                        cache: restoredCache,
                        parameters: params
                    )
                } else {
                    // MISS: full prefill.  We own the cache object so we can
                    // snapshot it (via copy()) immediately after init — the KVCache
                    // instances are classes and are mutated in-place by TokenIterator,
                    // so ownedCache already reflects the post-prefill state.
                    let ownedCache = makePromptCache(model: context.model, parameters: params)
                    let lmInput = LMInput(tokens: MLXArray(tokenIds))
                    iterator = try TokenIterator(
                        input: lmInput,
                        model: context.model,
                        cache: ownedCache,
                        parameters: params
                    )
                    // Snapshot at offset=N (all prompt tokens processed, before decode).
                    // Use toQuantized() to create independent data — .copy() uses
                    // MLXArray views that share the underlying buffer, so mutations
                    // during generation corrupt the "snapshot".
                    let snapshot: [any KVCache] = ownedCache.map { layer in
                        if let simple = layer as? KVCacheSimple {
                            return simple.toQuantized(groupSize: 64, bits: 4)
                        }
                        return layer.copy()
                    }
                    state.promptCache.save(tokenIds: tokenIds, kvCache: snapshot)
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

                if !cancelled, let chunk = detokenizer.next() {
                    _ = chunk.withCString { ptr in callback(ptr, callbackCtx) }
                }

                Stream().synchronize()
            }
            box.value = generated
        } catch {
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
                let ids = try context.tokenizer.applyChatTemplate(messages: messages)
                box.value = context.tokenizer.decode(tokenIds: ids)
            }
        } catch {
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
