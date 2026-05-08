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

                // --- Prompt KV cache ---
                // On a hit we restore the saved KV state (trimmed by 1 token),
                // then feed only the last prompt token to TokenIterator so that
                // prefill is a single decode step instead of N steps.
                // On a miss we run full prefill and snapshot the resulting cache
                // (before generation) for the next call with the same prompt.
                var iterator: TokenIterator
                if let entry = state.promptCache.lookup(tokenIds: tokenIds) {
                    // HIT: restore cache at offset N-1, run last token only.
                    let restoredCache = entry.kvCache.map { deepCopyCache($0) }
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
                        return deepCopyCache(layer)
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

    Task {
        do {
            try await state.container.perform(nonSendable: inputText) { context, txt in
                let tokenIds = context.tokenizer.encode(text: txt)
                guard !tokenIds.isEmpty else { return }

                let tokens = MLXArray(tokenIds.map { Int32($0) })
                let input = LMInput.Text(tokens: tokens.expandedDimensions(axis: 0))
                let output = context.model(input, cache: nil, state: nil)

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

                let hidden = matmul(
                    output.logits.asType(.float32),
                    fullWeight.asType(.float32))
                let seqAxis = hidden.ndim - 2
                let pooled = hidden.mean(axis: seqAxis).reshaped(-1)

                let norm = MLX.sqrt((pooled * pooled).sum())
                let normalized = norm.item(Float.self) > 0 ? pooled / norm : pooled
                MLX.eval(normalized)
                Stream().synchronize()

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
