#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle returned by mlx_model_load.
typedef struct MlxModelHandle MlxModelHandle;

// Load a model from a local MLX-format directory (safetensors + config.json).
// Returns NULL on failure. Caller must free with mlx_model_free.
MlxModelHandle* mlx_model_load(const char* model_path);

// Free a model handle and release all associated resources.
void mlx_model_free(MlxModelHandle* handle);

// Token callback: called once per decoded text chunk.
// Return 1 to continue generation, 0 to stop early.
typedef int (*MlxTokenCallback)(const char* token, void* user_data);

// Run synchronous (blocking) generation, calling `callback` for each token chunk.
// Params are passed individually to avoid C struct ABI issues across @_cdecl.
// Returns the number of completion tokens generated, or -1 on error.
int32_t mlx_generate(
    MlxModelHandle*  handle,
    const char*      prompt,
    uint32_t         max_tokens,
    float            temperature,
    float            top_p,
    uint32_t         seed,
    MlxTokenCallback callback,
    void*            user_data
);

#ifdef __cplusplus
}
#endif
