# Dual-GPU three-slot inference: V100 coding slot, P40 RE slot, iGPU batch slot

**Date:** 2026-09-23
**Status:** Design drafted in brainstorming; awaiting review before an implementation plan.
**Origin:** The Tesla P40 was pulled from this host and replaced by a Tesla V100 32GB
(`GV100GL [Tesla PG500-216]`, compute capability 7.0, 32494 MiB, at `01:00.0`). The P40 will
return as an eGPU once parts arrive. The goal is a coding agent on the V100 and PARE's
gemma-4-26b on the P40, running concurrently.

## Hardware and consumers

The machine currently has **one** CUDA device. `lspci` shows only the GV100GL and the AMD
Cezanne iGPU; the P40 is physically absent. "Both at the same time" is therefore the
post-eGPU state, and everything before Phase 4 is preparation.

Every slot already has a named consumer, which is what fixes the topology:

| Slot | Device | Model | Consumer |
|---|---|---|---|
| main | V100 (CUDA) | Qwen3.8-27B *(new)* | coding agent — opencode, or Claude Code |
| re | P40 (CUDA, eGPU) | gemma-4-26b-a4b | PARE, the RE agent |
| batch | AMD iGPU (Vulkan) | gemma-4-E4B | PARE's batch slot |

PARE pins gemma-4-26b in `PARE/pare/config.py:29`, with the comment *"Override BaseConfig's
Qwen default: gemma-4-26b handles tool-calling far more reliably for PARE's agentic loop."*
That is an empirical result the design should not discard. `PARE/.env.example` sets
`PARE_BATCH_MODEL=gemma-4-E4B-it-Q4_K_M`, which the iGPU slot already serves.

## Verified findings

Each of these was read out of the source or measured on the host, not inferred.

### F1 — The llama.cpp build targets Pascal only, and that feature-gates the V100

`/mnt/secondary/llama.cpp/build/CMakeCache.txt` has `CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=61`,
and the ggml-cuda compile flags are `--generate-code=arch=compute_61,code=[compute_61,sm_61]`.
The V100 runs today only because the driver JIT-compiles Pascal PTX to sm_70.

The cost is not merely JIT. ggml's capability checks key on `__CUDA_ARCH_LIST__` — what the
binary was *compiled* for — not on the hardware:

- `volta_mma_available()` (`ggml/src/ggml-cuda/common.cuh:318-320`) requires
  `ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA`.
- `ggml_cuda_should_use_wmma_fattn()` (`ggml/src/ggml-cuda/fattn-wmma-f16.cuh`) requires the same.
- `fast_fp16_available()` (`common.cuh:277-280`) returns **false** when
  `ggml_cuda_highest_compiled_arch(cc) == 610`.

With `__CUDA_ARCH_LIST__ = {610}`, `ggml_cuda_highest_compiled_arch(700)` returns 610, so all
three are false. The dedicated Volta flash-attention path at `ggml/src/ggml-cuda/fattn.cu:431-444`
is unreachable and selection falls through to the generic tile/vec kernels. ggml currently
believes this V100 has no fast FP16 — the P40's defining weakness, applied to a card with FP16
tensor cores.

**Consequence:** rebuild with `CMAKE_CUDA_ARCHITECTURES=61;70`. `61` must stay so the P40 keeps
working when it returns.

### F2 — The deployment has split provenance

`readelf -d /opt/llama/bin/llama-server` gives `RUNPATH: /mnt/secondary/llama.cpp/build/bin`.
The binary is a copy frozen at Apr 20; its ggml shared libraries resolve live from the build
tree. A rebuild in place would therefore (a) replace `.so` files that three running services
have mapped, and (b) leave an April binary loading September libraries — an ABI mismatch.

**Consequence:** build into a fresh directory and `cmake --install` to a versioned prefix; never
rebuild in place under running services.

### F3 — The checkout is five months stale

`HEAD` is `e365e658f`, dated 2026-04-19. Actual upstream is `1a679828f` (`git ls-remote`). The
local `origin/master` ref has not been fetched since April, so `git rev-list HEAD..origin/master`
reports 0 and is misleading. Since the rebuild is mandatory anyway, update first.

### F4 — KV cache sizing depends on attention shape, and the naive reading is wrong

Three shapes, each read from GGUF metadata:

- **full** — every layer holds KV for every token.
- **SWA** — `attention.sliding_window_pattern` marks windowed layers, which hold at most
  `attention.sliding_window` (+ padding) cells regardless of context. A constant, not per-token.
- **hybrid linear** — `ssm.*` keys plus `full_attention_interval = N`: only every Nth layer holds
  KV; the rest carry a fixed-size recurrent state.

Two traps found the hard way:

- `attention.head_count_kv` **can be a per-layer array**. gemma-4-26b is `[8,8,8,8,8,2, ...]` —
  8 on sliding-window layers, 2 on global layers. Reading index 0 overstated its cache 4x.
- Qwen3.5-35B-A3B looks like 40 full-attention layers and is actually 10.

Both corrections were confirmed against llama-server's own allocation log, which is the
authority. For gemma-4-26b at `n_ctx 65536` on the V100:

```
llama_kv_cache: size = 1280.00 MiB ( 65536 cells,  5 layers, 1/1 seqs)
llama_kv_cache: size =  300.00 MiB (  1536 cells, 25 layers, 1/1 seqs)   # SWA
```

and for Qwen3.5-35B-A3B at `n_ctx 32768`:

```
llama_kv_cache:        size = 640.00 MiB ( 32768 cells, 10 layers, 4/1 seqs)
llama_memory_recurrent: size = 251.25 MiB (     4 cells, 40 layers, 4 seqs)
```

**Consequence:** never set `CTX_SIZE` from a projection. Load the model, read the
`llama_kv_cache` line, then set it.

### F5 — gemma-4-26b fits the P40 at full native context

At 20 KiB/token plus ~300 MiB of fixed sliding-window cache, gemma-4-26b reaches its full
262144 training context in roughly 21.8 GiB at f16. The P40's 24GB accommodates that without
quantizing the cache. The current 65536 setting is far more conservative than necessary.

### F6 — Host RAM is 30 GiB, which rules out the large MoE candidates

`free -h` reports 30 GiB total, ~23 GiB available, shared with three llama-servers. Offloading
MoE experts to system RAM is not available here, so models needing it (Qwen3-Coder-Next at 80B,
Qwen3.8-Flash-Next at 180B on disk) are out at any quantization worth running. This host also
has a history of host-RAM OOM — see the `--cache-ram 0` rationale in
`systemd/llama-server.service`.

### F7 — The manager assumes one GPU and exactly two slots

- `manager/gpu.py:49` parses `lines[1]` of `nvidia-smi` output only; with two cards it silently
  reports the first.
- `manager/routing.py:23-28` hardcodes `main` then `batch`.
- `manager/config.py` carries `batch_*` fields rather than a slot collection.

### F8 — Live defect: `/status` reports stale startup state indefinitely

`/status` returned `loaded_model: null, healthy: false` for both slots while `:8081` served
gemma-4-26b and `:8083` served gemma-4-E4B.

Diagnosed, and narrower than it first appeared. `lifespan` (`manager/app.py:336-347`) probes each
slot once at startup. `reprobe_all_slots` (`manager/app.py:131`) exists — added by `e222feb` for
exactly this startup race, and its docstring describes this very symptom — but it is called only
from the chat-completions 409 path (`manager/app.py:478`). `/status` (`manager/app.py:404-414`)
is a passive reader of `to_status_dict()` and never triggers a probe.

Confirmed by experiment on the live server:

```
/status before any chat request : {'main': (None, False), 'batch': (None, False)}
one chat request for gemma-4-26b -> 200, served correctly
/status after                    : {'main': ('gemma-4-26b-a4b-it-q4_k_m', True),
                                    'batch': ('gemma-4-E4B-it-Q4_K_M', True)}
```

**So the routing path self-heals and only the monitoring surface lies.** This is a reporting bug,
not a routing bug, and therefore **not a prerequisite** for the slot generalization — it is an
independent small fix. It still matters: `/status` is what a human or a dashboard consults to
decide whether the fleet is healthy, and it currently reads as a total outage when nothing is
wrong.

### F9 — agent_core auto-swaps on 409

`agent_core/inference.py` keys on a `model_not_loaded` error body and issues `POST /swap` to
recover. With one model per slot this is benign, but a request for a model resident on no slot
will evict something. The three-slot design needs a guard.

## Decisions locked during brainstorming

- **Three slots**, not two: V100 + P40 + iGPU, each with its own consumer.
- **Coding model: Qwen3.8-27B.** Apache 2.0, 64 layers with `full_attention_interval 4`
  (16 full-attention layers), 262144 native context, released 2026-08-14. Chosen over
  Qwen3.5-35B-A3B (already on disk, cheaper KV, faster as a 3B-active MoE) on expected quality
  grounds; the tradeoff is generation speed.

  Two properties behind that tradeoff are **not yet confirmed locally** and should be checked in
  Phase 2 before the choice is treated as settled:
  - That it is **dense** (all 27B active per token) rather than MoE. This is what makes it
    slower than a 3B-active alternative, and therefore the whole basis of the tradeoff. Taken
    from the release write-up; the fetched `config.json` fields neither confirm nor exclude
    expert layers.
  - That its GGUFs declare `general.architecture = "qwen35"`, which this checkout already
    registers (`src/llama-arch.cpp`). If so it may load without the Phase 1 update; if not, it
    needs a llama.cpp new enough to know the architecture. Either way Phase 1 comes first.
- **PARE goes dark during the interim.** Once the V100 holds Qwen3.8-27B, gemma-4-26b has no
  home until the eGPU arrives. No fallback to gemma-4-E4B: an interim config would need
  unwinding, and a degraded PARE risks being misread as a model regression.
- **Rebuild before benchmarking.** Any measurement taken on the Pascal-only build describes
  JIT'd Pascal kernels and does not transfer.
- **GLM-4.7-Flash rejected.** Attractive on paper (MLA compresses KV to ~53 KiB/token) but
  carries an 18x downside if MLA does not engage — the fallback materializes
  `20 x (256+256)` per layer, ~940 KiB/token, which does not fit at 32k. The failure mode is
  reported in practice, and the one systematic OpenCode comparison found it failing structured
  tasks outright. Not worth the risk for this use case.

## Design

### Phase 1 — Rebuild and install with provenance

Mechanical; the implementation plan may carry exact commands.

1. Fetch and update `/mnt/secondary/llama.cpp` from `e365e658f` toward upstream. Record the
   chosen SHA.
2. Configure a **new** build directory, preserving the flags already in `CMakeCache.txt`
   (`GGML_CUDA=ON`, `GGML_CUDA_FA=ON`, `GGML_VULKAN=ON`, `GGML_CUDA_COMPRESSION_MODE=size`) and
   setting `CMAKE_CUDA_ARCHITECTURES=61;70`.
3. `cmake --install` to a versioned prefix, e.g. `/opt/llama/llamacpp-<short-sha>/`, so the
   binary and its ggml libraries ship together and `RUNPATH` resolves inside the prefix.
4. Repoint the three units at the new prefix. Keep the previous tree in place for rollback.
5. Record the commit SHA and the `sha256` of the installed binary alongside the install, so a
   deployed copy can be traced to a commit (F2).

**Verification, in order.** Services must be stopped before the install, not during it (F2).
After restart: all three units active, each backend answering `/v1/models`, and a
prompt-processing benchmark on gemma-4-26b compared against a baseline captured *before* the
rebuild. If that number does not move, the Volta path did not engage and the rebuild did not do
what F1 predicts — treat an unchanged benchmark as a failure, not a pass.

### Phase 1 build record (completed 2026-09-23)

```
llama.cpp b11136 = 057494f93f859308297cf8d21eee88e3fa17603c
installed to      /opt/llama/llamacpp-b11136   (bin/ + lib/, self-contained)
bin/llama-server  sha256 5a0337d681469286d446118e9dbe9de340252299c494c99b9bf24d17c5f6475c
replaced          sha256 2eac9d2ffe8e275ba98c4eeff663f288685688af49c090b51a10c5f2e6bf0a0a
                  kept at /opt/llama/bin/llama-server.b8851-e365e658f for rollback
/opt/llama/bin/llama-server is now a symlink into the versioned prefix, so the
units need no edit and a rollback is one symlink.
```

Measured, same model and flags on CUDA0 before and after:

| | prompt eval | generation |
|---|---|---|
| gemma-4-E4B, old `sm_61` binary | 2338.5 tok/s | 74.8 tok/s |
| gemma-4-E4B, new `sm_61;70` binary | 3181.5 tok/s | 96.6 tok/s |
| gemma-4-26b deployed, before | 865.8 tok/s | 81.1 tok/s |
| gemma-4-26b deployed, after | 1418.1 tok/s | 90.4 tok/s |

F1 predicted the Volta path was unreachable; enabling it is worth **+36% to +64%**
on prompt processing. The verification criterion was "if this number does not move, the
rebuild did not work" — it moved.

**Install trap worth recording.** CMake strips the build-tree RPATH on install and sets no
replacement, so the first staged install produced a binary that died with
`libllama-server-impl.so: cannot open shared object file`. Configuring
`CMAKE_INSTALL_RPATH='$ORIGIN/../lib'` fixes it, and `$ORIGIN` resolves correctly through the
`/opt/llama/bin/llama-server` symlink (tested). Any future rebuild must set this, or the
install is broken in a way that only appears when the services restart.

### F10 — b11136 reports full model paths where b8851 reported basenames

`/v1/models` and the chat-completion `model` field now return
`/opt/llama/models/gemma-4-26b-a4b-it-q4_k_m.gguf` rather than
`gemma-4-26b-a4b-it-q4_k_m.gguf`. Routing is unaffected — `display_name()` collapses a path to
its stem, which is exactly the case `2026-06-29-model-name-normalization-design.md` anticipated
— but any consumer that compares the returned `model` string against what it requested now sees
a different shape. Check PARE and the coding agent before assuming this is harmless.

### F11 — The deployed main unit has drifted from the repo

`/etc/systemd/system/llama-server.service` carries `--parallel 1` and `--temp 0.2`;
`systemd/llama-server.service` in this repo has `--parallel 2` and no `--temp`. The other three
units match the repo exactly. These are hand-edits made on the live host and never committed;
they were **not** made by this work and have been left in place. They need reconciling in one
direction or the other, and until they are, the repo does not describe what runs.

Note the Phase 1 benchmark above is unaffected: before and after used this same deployed unit.

### Phase 2 — Qwen3.8-27B on the V100

1. Download a GGUF. Quantization is chosen **after** step 2, not before.
2. Load it and read the `llama_kv_cache` line to establish the true bytes/token (F4). The
   projection to validate is 16 full-attention layers x 4 KV heads x (256+256) x 2 bytes.
3. Choose quantization and `CTX_SIZE` from that measurement against the V100's 32494 MiB,
   leaving room for the compute buffer. Record the reasoning next to the setting.
4. Update `config/llama-server.env`, correcting the stale comments that still describe CUDA0 as
   a Tesla P40 (`config/llama-server.env:10,17,27`).
5. gemma-4-26b leaves the V100 and PARE is expected to fail its health check until Phase 4.
   State this in `PARE`'s docs so the outage is not mistaken for a defect.

**Verification:** the model loads on the rebuilt binary; `/v1/models` reports it; a tool-calling
exchange through the manager succeeds end to end from opencode, not merely a 200 from the API.

### Phase 3 — Manager generalization to N slots

Cross-component, with an invariant that must hold across units and code, so this section
specifies interfaces and invariants rather than code. The implementer writes it against the
real files.

**Slot configuration.** Replace the `batch_*` fields in `manager/config.py` with a collection of
slot definitions, each carrying: name, host, port, env-file path, systemd unit, queue limit, and
a device identifier. Existing env-var names should keep working, so a deployed `manager.env`
does not have to change in the same step as the code.

**Routing.** `manager/routing.py:23-28` must iterate the configured slots instead of naming
`main` and `batch`. Priority is the order slots appear in configuration, with `main` first, so
today's main-then-batch behaviour is preserved exactly when two slots are configured. The
iteration order must come from an ordered structure, not from dict insertion order by accident.
The comparison semantics from
`2026-06-29-model-name-normalization-design.md` are unchanged. Two models resident on two slots
must route to their own slot with no dependence on dict ordering.

**Device pinning — the safety-critical invariant.** With two CUDA cards, `CUDA0` is assigned by
enumeration order and can change when the eGPU appears. A slot must be bound to a *physical
card*, not to an ordinal. Bind by GPU UUID (`nvidia-smi --query-gpu=uuid`) via
`CUDA_VISIBLE_DEVICES` in each unit, so each service sees exactly its own card. The invariant to
hold and to test: **restarting any service, in any order, with both cards present, never places
a model on the other card.** Failure here is silent — the wrong model loads on the wrong card
and appears to work — so this needs an explicit check, not an assumption.

**GPU reporting.** `manager/gpu.py:49` must report every card rather than `lines[1]`. The
`/status` shape changes; note whether any consumer depends on the current single-GPU shape
before changing it.

**Swap guard (F9).** Define what happens when a model resident on no slot is requested. The
current answer is 409, and agent_core reacts by calling `POST /swap`. Decide which slot, if any,
such a request may evict, and make that explicit rather than emergent.

**F8 is an independent fix, not a prerequisite.** Now that it is diagnosed, `/status` simply
needs to reflect reality — either by re-probing before reporting, or by a periodic background
probe that serves every reader rather than only this one endpoint. Prefer the latter if any
other surface depends on freshness; it stops the same class of bug recurring at the next passive
reader, rather than patching this one call site.

**Testing.** Routing and configuration are pure and should be unit-tested with three slots
configured. Every regression test must be verified failing against the pre-fix code. Prefer
assertions on relationships — "the number of slots reported equals the number configured" —
over assertions on literals, which break on the next legitimate change.

### Phase 4 — eGPU bring-up

Deferred until the hardware arrives; scope recorded so it is not lost.

- Confirm both cards enumerate and that the UUID pinning from Phase 3 holds across reboots.
- Bring up the `re` slot on the P40 with gemma-4-26b. Per F5 it can run at its full 262144
  context at f16 in roughly 21.8 GiB; confirm against the `llama_kv_cache` line rather than
  against that projection.
- Restore PARE's configuration and confirm a full RE loop end to end.
- Re-examine eGPU link bandwidth. Weights are loaded once, so the enclosure's link mainly costs
  load time, but this should be measured rather than assumed.

## Out of scope

- Any change to PARE, agent_core or the code graph beyond restoring PARE's model.
- The code graph itself, which is forthcoming and independent of this work.
- `config/manager.env.orig` — a stale March backup (`PORT=11343`, missing every Phase-B key),
  untracked in the working tree. Not created by this work and not removed by it.

## Open questions

1. Which Qwen3.8-27B quantization, settled by the Phase 2 measurement rather than by projection.
2. Whether Qwen3.8-27B is dense, and whether its GGUF architecture is `qwen35` — both taken from
   published material, both checked in Phase 2, and the first underpins the model choice.
3. Whether any consumer depends on the single-GPU `/status` shape before Phase 3 changes it.
4. For F8, whether to re-probe inside `/status` or add a periodic background probe. The periodic
   probe serves every future reader; the per-endpoint fix is smaller. Decide in the plan.
