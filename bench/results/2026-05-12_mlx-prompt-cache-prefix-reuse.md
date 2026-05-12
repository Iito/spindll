# MLX Prompt Cache — Prefix Reuse

**Date:** 2026-05-12
**Model:** `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (Metal, MLX backend)
**Harness:** `bench/chat_prefix_bench.py` against `spindll serve` (HTTP, `--budget 8G`)
**Workload:** ~1800-token system prompt + 1 user turn; 6 reps. Each rep's system
prompt carries a rep-unique marker at the front so nothing is pre-cached.

- **cold** = turn 1 of the conversation → full prefill of the whole prompt.
  (This is what the *old* prompt cache did on every turn of a conversation,
  because each turn's token sequence differs from the last.)
- **warm** = turn 2 of the same conversation → prefix hit; only the new turn
  (`assistant reply` + `user 2` + assistant header, ≈ 25 tokens) is prefilled.

## TTFT (ms)

| series | mean | median | min | max |
| --- | --- | --- | --- | --- |
| cold (full prefill)  | 13638 ms | 13682 ms | 13343 ms | 13818 ms |
| warm (prefix reuse)  |   551 ms |   544 ms |   519 ms |   615 ms |

**TTFT (median): 13682 ms → 544 ms — −96.0%** (≈ 25× faster).

## Notes

- Absolute cold TTFT is unusually high here: the test box was under heavy load
  (concurrent builds), so MLX prefill ran at ≈ 130 tok/s instead of the usual
  500–2000 tok/s. The *ratio* is the headline — prefix reuse removes essentially
  all of the prefill cost when the prompt prefix is already resident.
- The residual ≈ 540 ms in the warm case is the rest of the request pipeline on
  this loaded box (chat-template application, the ≈ 25-token suffix prefill, the
  q8 snapshot deep-copy, the FFI round-trip, and 4 decode steps at ≈ 77 ms/tok).
  None of that is prefill — the prefill itself drops to near zero.
- The standard `cargo run -- bench <model>` command exercises the *raw* generate
  path (`mlx_generate`), which has no prompt cache, so it cannot measure this.
  The prompt cache only applies to the chat path (`mlx_chat_generate`), reached
  via the HTTP/gRPC chat endpoints — hence the dedicated harness above.
- Same run also exercised: 8-bit snapshot for the freshest entry, demote-to-4-bit
  for superseded entries, and the `canTrimPromptCache` guard (Mamba/SSM hybrids
  are not cached). No crashes; cache-hit output matches cold output for repeated
  prompts under greedy decoding.
