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

// Retains the ModelContainer across FFI calls.
private final class ModelState: @unchecked Sendable {
    let container: ModelContainer
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

    // Tune MLX caches once per load. Setting these per-generate caused
    // re-acquire churn; doing it here amortises across all requests.
    Memory.cacheLimit = 64 * 1024 * 1024

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

            // Prepare input through the model's own processor (handles chat templates etc.)
            let lmInput = try await state.container.prepare(
                input: UserInput(prompt: promptStr)
            )

            // Run the token loop synchronously inside actor isolation. This
            // avoids the per-token Sendable hop of AsyncStream<Generation>
            // and matches the perf profile of Apple's `llm-tool` CLI.
            var generated: Int32 = 0
            try await state.container.perform(nonSendable: lmInput) { context, lmInputLocal in
                var iterator = try TokenIterator(
                    input: lmInputLocal,
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

                var stopped = false
                while let tokenId = iterator.next() {
                    if stopTokenIds.contains(tokenId) { break }
                    generated += 1
                    detokenizer.append(token: tokenId)
                    if stopped { continue }
                    if let chunk = detokenizer.next() {
                        let shouldContinue = chunk.withCString { ptr in
                            callback(ptr, callbackCtx)
                        }
                        if shouldContinue == 0 { stopped = true }
                    }
                }

                // Flush any partial-UTF-8 bytes the detokenizer was holding
                // back (e.g. when the loop hits maxTokens mid-codepoint).
                if !stopped, let chunk = detokenizer.next() {
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
