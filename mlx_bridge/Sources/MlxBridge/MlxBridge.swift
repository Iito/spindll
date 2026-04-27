import Foundation
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

/// Blocking generation: iterates the AsyncStream<Generation> and calls `callback`
/// once per .chunk. Returns generationTokenCount on success, -1 on error.
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

            var stopped = false
            let stream = try await state.container.generate(
                input: lmInput,
                parameters: params
            )

            for await generation in stream {
                switch generation {
                case .chunk(let text):
                    if stopped { continue }
                    let shouldContinue = text.withCString { ptr in
                        callback(ptr, callbackCtx)
                    }
                    if shouldContinue == 0 { stopped = true }

                case .info(let info):
                    box.value = Int32(info.generationTokenCount)

                case .toolCall:
                    break // not supported in PoC
                }
            }

            if box.value == nil { box.value = -1 }
        } catch {
            box.value = -1
        }
        sema.signal()
    }
    sema.wait()

    return box.value ?? -1
}
