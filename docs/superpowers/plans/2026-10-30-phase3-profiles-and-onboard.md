# Phase 3: Per-Model Profiles and `onboard` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `.args` sidecars, `SLOT_<NAME>_MODELS`, and the in-place env-file regex with one TOML profile per model. Every swap renders the slot's whole model configuration from the profile and writes it atomically, so no flag can carry over from the previous model. `onboard` verifies each (model, slot) pair with a trial load.

**Architecture:**
- **Profiles:** `config/models/<stem>.toml` is parsed and validated by a new module, `manager/profiles.py`.
- **Swaps:** a swap renders the profile's section for the slot into `MODEL_ARGS` plus any per-model env vars. It writes the result atomically to `/var/lib/llama-manager/<slot>.env`, which the slot's unit loads as an `EnvironmentFile`. `ExecStart` keeps only the slot-owned flags.
- **Roster:** the profiles are the roster. `/swap` accepts only verified (model, slot) pairs, unless `trial` is set, which is how `onboard` loads a profile that isn't verified yet.

**Tech Stack:** Python 3.12.3 (`tomllib` is in the stdlib, so this adds no dependency), FastAPI 0.135.2, llama.cpp build 11136 (`057494f93`), and systemd unit files.

**Spec:** `docs/design-review-2026-10-29.md`, the "Profiles" section: Q13, Q25, Q29–Q34 and Q42.

## Global Constraints

- **Test command:** `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/ -q -p no:cacheprovider`.
- **Live ports:** tests never touch live ports 8080–8084 or 11434.
- **Modules:** new modules go flat in `manager/`. `deploy-manager.sh:143` ships only `manager/*.py`.
- **Profile location:** `config/models/<stem>.toml`, where `<stem>` is the GGUF's file name without `.gguf`. Profiles deploy to `/etc/llama/models.d/`, and GGUFs stay in `/opt/llama/models/` (Q31).
- **Generated model env files:** `/var/lib/llama-manager/<slot>.env`, owned by `_llama-mgr` (Q32).
- **Required keys:** `ctx_size`, `n_gpu_layers`, and every key in `SAMPLING_KEYS`. A missing key is a validation error. A flag a profile doesn't mention is absent, never inherited (Q33).
- **`ExecStart` keeps only slot-owned flags:** `--host ${HOST} --port ${PORT} --device ${DEVICE} --parallel 1 --cache-ram 0 $MODEL_ARGS` (Q29).
  - `--parallel 1` is Q10.
  - `--cache-ram 0` is a host-RAM workaround, not a property of any model.
  - `$MODEL_ARGS` stays **unbraced**: systemd word-splits `$VAR` and passes `${VAR}` as one argument.
- **Client API:** `/v1/models` ids, `/status` `slots.*.loaded_model`/`healthy`/`swapping`, and `POST /swap {model, target}` keep their meaning. They're what agent_core (`agent_core/commands/_builtin_impls.py:289-359`) and PAL (`pal/commands/domain.py:586`, `pal/agent.py:627`) read. Responses may only gain fields.

## How this phase runs (medium-heavy)

- **Claude commits each task's tests before the task starts**, including any rewrite of an existing test the task changes, and checks that every new test fails against the code before the task. **Qwen never edits a test.** If a test looks wrong, stop and report it.
- **After Qwen finishes a task**, the user runs the opencode mutation agent on the task's module against the task's tests. Claude reviews the diff and the surviving mutants before the next task.
- **Tasks 1–3 carry no code.** Their Interfaces blocks give signatures and invariants. The tests are the contract.

## Implementer Rules (Qwen)

1. Do exactly one task, then stop and report "Done" plus a **Deviations** list.
2. Never edit a test file. If one is wrong, stop and report it.
3. Stop and report if a step's Expected result doesn't happen.
4. Never run sudo, systemctl, deploy scripts or `git push`, and never touch ports 8080–8084 or 11434.
5. If anything in this plan turns out to be wrong, deviating is correct; say so in Deviations.
6. Commits stage explicit paths and end with the repo's attribution lines.

## Review Focus

Each item names the task whose tests pin it:

1. **A profile value containing whitespace, a newline or a quote** would inject extra arguments or extra env lines into the generated file. Validation must reject it. → Task 1, `test_values_that_could_inject_into_the_env_file_are_rejected`.
2. **`ctx_size = true`**: TOML booleans are Python `int`s. → Task 1, `test_booleans_are_not_integers`.
3. **A profile deleted from the repo but left in `/etc/llama/models.d/`** would keep serving after it was removed from the repo. → Task 5, `test_check_reports_a_profile_left_behind_on_the_host`.
4. **A write failure during a swap:** the old model env file must stay intact, with no restart. → Task 2, `test_a_failed_write_leaves_the_old_file_and_does_not_restart`.
5. **A trial swap to the model that's already loaded** must really restart with the profile's flags, not answer "already loaded". Otherwise onboarding the two live models would verify nothing. → Task 3, `test_a_trial_swap_restarts_even_when_the_model_is_loaded`.

## Decisions made while planning (beyond the spec)

- **An invalid profile doesn't stop the manager.** Startup leaves it out, logs it at ERROR, and reports it in `/status` as `profile_errors`. Separately, `deploy-manager.sh` refuses to ship any profile that fails validation (exit 5), so the error surfaces at deploy time, not at 3 a.m. Q33 calls it "a validation error at manager startup", and one bad file taking all inference down seemed worse than reporting it.
- **The full sampling set** is `temp, top_k, top_p, min_p, presence_penalty, frequency_penalty, repeat_penalty`. These are every sampler llama.cpp b11136 applies with a non-neutral default, plus the three penalties.
- **Bootstrap profiles keep today's effective values**, including llama.cpp's defaults wherever a sidecar was silent. Qwen3.6 has no `--min-p` in its sidecar, so it runs at the default `0.05`; its card says `0.0`. Changing that is tuning, which belongs in the benchmarking review, not in this refactor.
- **Profile `env` keys** may not be `HOST`, `PORT`, `DEVICE`, `CUDA_VISIBLE_DEVICES` or `MODEL_ARGS`, and may not start with `LLAMA_ARG_`. llama-server reads `LLAMA_ARG_*` env vars as flags, which would let a profile set a flag that `MODEL_ARGS` doesn't show.
- **The unit loads the model env file first and the slot env file second**, so the slot wins any conflict. Validation already rejects the obvious conflicts; this ordering is the backstop.

---

## Branch `phase3/profiles` (worktree `/mnt/secondary/inference-server-wt-phase3`)

### Task 1: `manager/profiles.py`: parse, validate, render

**Files:** Create `manager/profiles.py`. Tests: `tests/test_profiles.py`, committed by Claude.

**Profile format:**

```toml
home = "main"                      # required; must name one of the sections below

[slots.main]                       # one section per slot this model may run on
ctx_size = 131072                  # required int > 0
n_gpu_layers = -1                  # required int >= -1
flags = ["--jinja", "--ctx-checkpoints", "4"]   # optional list of single tokens

[slots.main.sampling]              # required: exactly SAMPLING_KEYS
temp = 1.0
top_k = 20
top_p = 0.95
min_p = 0.0
presence_penalty = 0.0
frequency_penalty = 0.0
repeat_penalty = 1.0

[slots.main.env]                   # optional per-model env vars
# EXAMPLE_KEY = "value"

[slots.main.verified]              # written by scripts/onboard.py; absent = not verified
date = "2026-10-31"
llama_cpp_build = 11136
llama_cpp_commit = "057494f93"
```

**Interfaces (Produces):**
- `SAMPLING_KEYS: tuple[str, ...] = ("temp", "top_k", "top_p", "min_p", "presence_penalty", "frequency_penalty", "repeat_penalty")`. Each renders as a flag: `--temp`, `--top-k`, `--top-p`, `--min-p`, `--presence-penalty`, `--frequency-penalty`, `--repeat-penalty`.
- `RESERVED_FLAGS: frozenset[str]`: flags a profile's `flags` list may not contain, compared on the part before any `=`. The set:
  - set by the slot: `--host`, `--port`, `--device`, `-dev`, `--parallel`, `-np`, `--cache-ram`, `-cram`;
  - set by the renderer: `--model`, `-m`, `--ctx-size`, `-c`, `--n-gpu-layers`, `--gpu-layers`, `-ngl`;
  - every sampling flag above, plus `--temperature`.
- `RESERVED_ENV: frozenset[str] = {"HOST", "PORT", "DEVICE", "CUDA_VISIBLE_DEVICES", "MODEL_ARGS"}`. Keys starting with `LLAMA_ARG_` are rejected too.
- `class ProfileError(ValueError)`. Its message names the profile and the offending key.
- `@dataclass(frozen=True) class Verified`: `date: str`, `llama_cpp_build: int`, `llama_cpp_commit: str`.
- `@dataclass(frozen=True) class SlotSection`:
  - `slot: str`, `ctx_size: int`, `n_gpu_layers: int`;
  - `sampling: tuple[tuple[str, int | float], ...]`, in `SAMPLING_KEYS` order;
  - `flags: tuple[str, ...]`;
  - `env: tuple[tuple[str, str], ...]`, sorted by key;
  - `verified: Verified | None`.
- `@dataclass(frozen=True) class Profile`: `name: str`, `home: str`, `sections: Mapping[str, SlotSection]`.
  - `def section(self, slot: str) -> SlotSection | None`.
  - `def verified_slots(self) -> tuple[str, ...]`, in section order.
- `def parse_profile(name: str, data: dict, slot_names: Collection[str]) -> Profile`. Pure. Raises `ProfileError` on any rule below.
- `def load_profile(path: str, slot_names: Collection[str], models_dir: str) -> Profile`. Runs `tomllib` and `parse_profile`, with `name` set to the file stem. It also requires `<models_dir>/<name>.gguf` to exist with a size above 0.
- `@dataclass class ProfileSet`: `profiles: dict[str, Profile]`, keyed by `names.match_key(name)`; `errors: dict[str, str]`, keyed by file stem.
  - `def get(self, model: str) -> Profile | None`. Case- and `.gguf`-insensitive, via `names.match_key`.
  - `def onboarded(self) -> list[Profile]`: profiles with at least one verified section, sorted by name.
  - `def home_models(self) -> dict[str, list[str]]`: slot to sorted profile names.
- `def load_profiles(profiles_dir: str, slot_names: Collection[str], models_dir: str) -> ProfileSet`. Loads every `*.toml` and never raises.
  - A bad file goes into `errors`.
  - A missing directory gives an empty set with `errors[profiles_dir]` set.
  - Two files whose names collide under `match_key` both go into `errors`.
- `def render_model_env(profile: Profile, section: SlotSection, models_dir: str) -> str`. The exact output:
  - line 1: `# Generated by llama-manager from profile <name> [slots.<slot>]. Every swap rewrites this file; do not edit.`
  - line 2: `MODEL_ARGS=` followed by these tokens, space-joined, in order:
    1. `--model <models_dir>/<name>.gguf`;
    2. `--ctx-size <n>`, then `--n-gpu-layers <n>`;
    3. each sampling flag in `SAMPLING_KEYS` order, with ints via `str()` and floats via `repr()`;
    4. then `flags`, as given.
  - then one `KEY=VALUE` line per `env` entry, sorted by key;
  - a trailing newline.
- CLI, `python -m manager.profiles`:
  - `check <dir> --slots a,b --models-dir <dir>` prints each error and exits 1 if there are any. Otherwise it prints `N profiles OK` and exits 0.
  - `render <dir> <model> <slot> --slots a,b --models-dir <dir> [--write PATH]` prints the rendered file. It works for unverified sections. With `--write`, it writes the file atomically through `manager.swap.write_model_env` from Task 2. Until Task 2 lands, `--write` may be left out: Task 2's tests cover it.

**Validation rules:** each rule is pinned by a test in `tests/test_profiles.py`.
- **Profile names:** `[A-Za-z0-9._-]+`.
- **Allowed keys:** top-level keys are exactly `home` and `slots`. Section keys are a subset of `ctx_size, n_gpu_layers, flags, sampling, env, verified`. An unknown key anywhere is an error; this is the typo guard.
- **`home`** must name a section. Every section name must be in `slot_names`, and there must be at least one section.
- **Integers:** `ctx_size` is an int above 0, `n_gpu_layers` an int of at least -1, and `top_k` an int of at least 0. Other sampling values are int or float. **`bool` is never accepted where a number is expected.**
- **Tokens and values:** every flag token and env value is a non-empty str with no whitespace, no control character, and none of these: double quote, single quote, backslash, `$`, backtick. Sampling values must be finite (TOML allows `nan` and `inf`). Env keys match `[A-Z_][A-Z0-9_]*` and aren't reserved.
- **`verified`**, when present, has exactly `date` (`YYYY-MM-DD` str), `llama_cpp_build` (int) and `llama_cpp_commit` (str).

- [ ] **Step 1:** Run `tests/test_profiles.py`. Expected: everything fails, because `manager.profiles` doesn't exist yet.
- [ ] **Step 2:** Implement `manager/profiles.py` per the Interfaces block.
- [ ] **Step 3:** Run the full suite. Expected: everything passes.
- [ ] **Step 4:** Commit: `feat(profiles): parse, validate and render per-model profiles (Q13, Q29, Q33)`.

### Task 2: Swaps write the generated model env file; units load it

**Files:**
- Modify: `manager/swap.py`. Delete `sampling_args_for_model` and `update_env_file`.
- Modify: `manager/slot_config.py`, adding `SlotConfig.model_env_file`.
- Modify: `manager/slots.py`, adding `SlotState.model_env_file`.
- Modify: `manager/app.py`, only where it builds `SlotState` and calls `swap_to`.
- Modify: `systemd/llama-server.service` and `systemd/llama-server-egpu.service`.
- Modify: `config/llama-server.env` and `config/llama-server-egpu.env`. Remove `MODEL_PATH`, `SAMPLING_ARGS`, `CTX_SIZE` and `N_GPU_LAYERS` along with their notes; the notes move into the profiles in Task 6.
- Tests, committed by Claude: `tests/test_model_env.py` and `tests/systemd_expand.py`. The latter is a test helper that expands a unit's `ExecStart` the way systemd does. Claude also rewrites `tests/test_swap.py` and any test that relied on `.args` sidecars.
- `tests/conftest.py` gains a `profiles_dir` holding verified profiles for `test-model-q4` and `test-model-q8` on both slots, `home = "main"`, and `test_config` points `profiles_dir` at it. This lands in this task's test commit, because from this task on a swap renders from a profile.

**Interfaces:**
- `def write_model_env(path: str, text: str) -> None`, in `manager/swap.py`. Atomic: write a temp file in the same directory, `fsync`, `chmod 0o644`, then `os.replace`. On any error, unlink the temp file and re-raise.
- `ModelSwapper.swap_to(self, model_name: str, env_text: str) -> bool`:
  1. `write_model_env(self._slot.model_env_file, env_text)`;
  2. `restart_llama_server()`;
  3. `wait_for_health()`.

  If the write raises, no restart happens and the exception propagates. `ensure_model_on_slot` already turns a raising swap into a failed swap.
- `SlotConfig.model_env_file: str`. `build_slots` sets it to `f"{getenv('MODEL_ENV_DIR') or '/var/lib/llama-manager'}/{name}.env"`. `SlotState.model_env_file` carries it.
- Until Task 3 lands, `ServerState.ensure_model_on_slot` keeps its signature and builds `env_text` with `render_model_env`. The profile comes from a `ProfileSet` loaded from `config.profiles_dir`. Add `ManagerConfig.profiles_dir: str = "/etc/llama/models.d"`, read from env `PROFILES_DIR`.
- Unit `[Service]`, in this order:
  1. `Environment=DEVICE=CUDA0`;
  2. `EnvironmentFile=/var/lib/llama-manager/<slot>.env`;
  3. `EnvironmentFile=/etc/llama/llama-server[-egpu].env`;
  4. `ExecStart=/opt/llama/bin/llama-server --host ${HOST} --port ${PORT} --device ${DEVICE} --parallel 1 --cache-ram 0 $MODEL_ARGS`.

  `<slot>` is `main` or `egpu`. Keep the units' existing comments where they're still true. Comments about removed keys move to the profiles.

**Pinned by the committed tests (2026-10-31, checked against a reference implementation; ten hand mutants all killed):**
- `SlotConfig.model_env_file: str | None = None` and `SlotState.model_env_file: Optional[str] = None` (last field). The default keeps every hand-built `SlotConfig`/`SlotState` in the suite working; `build_slots` always sets it.
- `write_model_env` calls `os.fsync`, `os.chmod` and `os.replace` through the `os` module (the tests patch `os.replace` and `os.fsync`). The temp file lives in the target's directory, and the target is never opened for writing.
- `ServerState.__init__` loads `self.profiles = load_profiles(config.profiles_dir, [slot names], config.models_dir)` and logs each `errors` entry at ERROR.
- `ensure_model_on_slot` looks up `self.profiles.get(model_name)` and its `section(slot_name)` in place of `model_path`. A missing profile or section takes the existing missing-file tail (unhealthy, drained queue, `False`) and never calls `swap_to`. On success it calls `swap_to(profile.name, render_model_env(profile, section, config.models_dir))` and `mark_swapped(profile.name)`.
- `model_path` and `list_models` stay until Task 3; the chat routes still use them.
- Test files in this task's test commit: `tests/test_model_env.py` (new), `tests/systemd_expand.py` (new), `tests/test_swap.py` (rewritten), `tests/conftest.py` (`profiles_dir`, `model_env_dir`, `write_test_profiles`, non-empty test GGUFs, model keys dropped from the fake slot env files, `three_slot_config` gets its own profiles with an `aux` section). Fakes in `tests/test_endpoints.py` keep their old parameter names; they take two positional arguments, so they still match.

- [ ] **Step 1:** Run `tests/test_model_env.py tests/test_swap.py`. Expected: they fail. The discriminating test is `test_a_swap_passes_exactly_the_profiles_flags_to_llama_server`: it expands the **repo's** unit files, and today's `ExecStart` hardcodes `--ctx-checkpoints 4 --jinja`, which no profile in the test declares.
- [ ] **Step 2:** Implement per the Interfaces block.
- [ ] **Step 3:** Run the full suite. Expected: everything passes.
- [ ] **Step 4:** Commit: `feat(swap): render the slot's model env from its profile and write it atomically (Q29, Q32)`.

### Task 3: The roster comes from profiles

**Files:**
- Modify: `manager/app.py`, `manager/slot_config.py` and `manager/routing.py`.
- Delete: `SlotConfig.models`, `_parse_models`, `_check_one_home_per_model`, `routing.home_slot`, `ServerState.model_path` and `ServerState.list_models`.
- Tests, committed by Claude: `tests/test_profile_roster.py`, plus rewrites of the affected tests in `tests/test_endpoints.py`, `tests/test_slot_config.py`, `tests/test_routing.py` and `tests/test_consumer.py`. The test profiles come from Task 2.

**Interfaces:**
- `ServerState.__init__(self, config: ManagerConfig, profiles: ProfileSet | None = None)`. When `profiles` is `None`, it calls `load_profiles(config.profiles_dir, [s.name for s in config.slots], config.models_dir)`. It exposes `self.profiles` and logs each entry in `profiles.errors` at ERROR.
- `ServerState.ensure_model_on_slot(self, slot_name: str, model_name: str, *, force: bool = False) -> bool`.
  - It renders from `self.profiles.get(model_name).section(slot_name)`.
  - Without `force`, a healthy slot already holding the model returns True with no restart. With `force`, it always swaps.
  - A missing profile or section is a refusal, not a failed swap: log it at ERROR and return False, leaving the slot's health, loaded model and queue alone. Nothing was stopped, so the old model is still serving. (This replaces Task 2's pinned "missing-file tail", which this plan specified and which was wrong: it drained requests that the still-running model could serve. Deviating from Task 2's code here is correct.) After this task both paths are defensive, since `/swap` and chat refuse such models first.
- **Chat requests:** `_validate_and_route` answers `model_not_found` (404, message `Model not found: <name>`) when `server.profiles.get(name)` is `None`.
- **`POST /swap` body:** `{"model": str, "target": str | None, "trial": bool = false}`. It checks, in order:
  1. Existing body checks (`invalid_body`, `missing_model`).
  2. `trial` present and not a bool: 400 `invalid_trial`.
  3. No profile: 404 `model_not_found`, message `no profile for model '<m>'; add config/models/<m>.toml`.
  4. `target` absent or null: use `profile.home`. A target that isn't a slot: existing 400 `invalid_target`.
  5. Profile has no section for the target: 400 `slot_not_allowed`, message `model '<m>' has no [slots.<t>] section; it may run on: <comma list>`.
  6. Section not verified and `trial` false: 400 `not_verified`, message `model '<m>' is not verified on slot '<t>'; run scripts/onboard.py <m> --slot <t>`.
  7. Not `trial`, slot healthy and the model loaded: 200, as today.
  8. Otherwise: 202 `swapping`. The background task calls `ensure_model_on_slot(target, model, force=trial)`.

  `no_home_slot` disappears: every profile has a home.
- **`GET /v1/models`:** one entry per `profiles.onboarded()`. Each entry keeps the existing four keys and adds:
  - `home_slot: str`;
  - `verified_slots: list[str]`;
  - `loaded_on: list[str]`: slots whose `loaded_model` matches, via `same_model`.
- **`GET /status`:** `slots.<name>.models` becomes `profiles.home_models().get(name, [])`. A new top-level `profile_errors: dict[str, str]`.

**Pinned by the committed tests (2026-10-31, checked against a reference implementation; hand mutants and the Task 2 mutation survivors all killed):**
- `"trial": null` is `invalid_trial`, unlike `"target": null`, which means absent. Any non-bool is rejected, ints included.
- A profile that failed validation (it is in `profile_errors`) is not in the roster: chat and `/swap` answer 404.
- `slot_not_allowed` comes before the `trial` check, so `trial` never runs a model on a slot it has no section for.
- A leftover `SLOT_<NAME>_MODELS` line in `manager.env` is ignored, not an error: the host keeps those lines until cutover step 7.
- `home_models()` and the roster are read once at startup; a profile edited later is picked up on restart.
- Test files in this task's test commit: `tests/test_profile_roster.py` (new); `tests/test_model_env.py` (the two refusal tests now also check health, queue and log; a new test that the slot env file's `DEVICE=` beats the unit default, which already passes); the home-map, `SLOT_*_MODELS` and `model_path` tests deleted from `tests/test_endpoints.py`, `tests/test_routing.py` and `tests/test_slot_config.py`. `tests/test_consumer.py` needed no change.

- [ ] **Step 1:** Run the suite. Expected: the tests Claude committed for this task fail, and nothing else does.
- [ ] **Step 2:** Implement per the Interfaces block.
- [ ] **Step 3:** Run the full suite. Expected: everything passes.
- [ ] **Step 4:** Commit: `feat(roster): profiles are the roster; /swap takes verified pairs or trial (Q13, Q30)`.

### Task 4: `scripts/onboard.py`

**Files:** Create `scripts/onboard.py`. Tests: `tests/test_onboard.py`, committed by Claude. They use `httpx.MockTransport`, so they need no network.

**Interfaces:**
- `def onboard(model: str, slot: str | None, *, client: httpx.Client, repo_profiles: str, deployed_profiles: str, models_dir: str, llama_version: Callable[[], tuple[int, str]], today: Callable[[], str], timeout: float = 900, poll: float = 2) -> int`. It returns an exit code. `main()` builds the real dependencies:
  - `client` from `--manager`, default `http://127.0.0.1:11434`;
  - `llama_version` by running `/opt/llama/bin/llama-server --version` and parsing `build (\d+), commit (\w+)`.
- **Steps and exit codes:**
  - **2:** the repo profile `config/models/<model>.toml` is invalid. Slot names come from `GET /status`.
  - **3:** the target slot (`slot`, or `home` by default) is already verified. Print: delete `[slots.<s>.verified]` to re-verify.
  - **4:** the repo file's sha256 differs from `<deployed_profiles>/<model>.toml`. Print: deploy first. The manager must be loading the exact file being verified.
  - **The trial:**
    1. Record `previous = /status slots[s].loaded_model`.
    2. `POST /swap {model, target: s, trial: true}`, expecting 202.
    3. Poll `/status` until the slot is healthy, not swapping, and has the model loaded. The deadline is `timeout`.
    4. `POST /v1/chat/completions {model, messages: [{"role": "user", "content": "Reply with the single word: ready"}], max_tokens: 256}`, expecting 200 with non-empty `choices`.
  - **Swap back:** if `previous` is set and isn't the same model, `POST /swap {model: previous, target: s, trial: true}` and poll the same way. This runs whether or not the trial succeeded.
  - **1:** the trial or the swap back failed. Nothing is written.
  - **0:** append to the repo profile:

    ```
    \n[slots.<s>.verified]\ndate = "<today>"\nllama_cpp_build = <n>\nllama_cpp_commit = "<c>"\n
    ```

    Then re-load the file, confirm `<s>` is now in `verified_slots()`, and print: commit and deploy.

**Pinned by the committed tests (2026-10-31, checked against a reference implementation; seventeen hand mutants all killed).** Three points correct the Interfaces block above, which this plan got wrong; deviating from it there is correct:
- **When a swap has landed (correction).** `/swap` returns 202 before its background task takes the slot's lock, so the first polls can show the slot exactly as before. When the model is already loaded (onboarding the two live models in Task 7) that looks like success, and the chat would hit the old process. Read the slot's `last_swap_utc` just before each `POST /swap`; the swap has landed only when `/status` shows a *different* `last_swap_utc`, `healthy`, not `swapping`, and `loaded_model` matching via `names.same_model`.
- **When a swap has failed.** Not `swapping` and not `healthy`, once `swapping` has been seen or if the slot was healthy before the POST. A slot that was already down is not a failed trial. Failure is recognised when `/status` shows it, not by running out `timeout`. A non-202 from `/swap` fails at once, with no polling.
- **Exit 5 (new).** The model is already loaded on another slot. Chat goes to the first configured slot holding the model (`routing.resolve_slot`), so the chat would be answered there and verify nothing about the target. Checked before any swap.
- **Default `--manager` (correction).** `manager_url_from_env(text: str) -> str` builds `http://<HOST>:<PORT>` from `/etc/llama/manager.env`; `HOST=0.0.0.0` or absent means `127.0.0.1`, `PORT` absent means `11434`. On this host the manager binds `192.168.1.14` and `127.0.0.1:11434` refuses connections, so the plan's default would fail every run.
- Exit 2 also covers a missing repo profile, a slot the manager doesn't have, and a slot the profile has no section for.
- Both swaps (trial and swap back) send `"trial": true` and use the same landed/failed rules. The swap back runs after a failed swap, a failed chat or a passed trial alike; a failed swap back is exit 1 and writes nothing.
- The chat body is exactly `{"model", "max_tokens": 256, "messages": [...]}`; `choices: []` is a failure.
- `parse_llama_version(text: str) -> tuple[int, str]`. The real `--version` output also prints a log line first (`version: 0.4.1-dev (build 11136, commit 057494f93)` on the host today); `main()` passes it stdout plus stderr.
- The script puts the repo root on `sys.path`, so `python3 scripts/onboard.py` runs from any directory with no `PYTHONPATH`.
- Untested: `main()`'s `httpx.Client` needs a long timeout (a cold load and the first chat can take minutes); use at least 600 s.
- Test file in this task's test commit: `tests/test_onboard.py` (new).

- [ ] Steps: run the tests (they fail) → implement → full suite passes → commit `feat(onboard): verify a profile with a trial load (Q25)`.

### Task 5: Deploy ships profiles; doctor checks the new layout

**Files:** Modify `scripts/deploy-manager.sh`, `scripts/doctor.sh`, `scripts/setup.sh` and `config/manager.env`.
- In `config/manager.env`: drop `SLOT_*_MODELS` together with the "Home slot per model" comment above them (it cites `routing.home_slot`, deleted in Task 3), and add `PROFILES_DIR=/etc/llama/models.d` and `MODEL_ENV_DIR=/var/lib/llama-manager`. Also fix its comment at line 15, which still says the manager updates `MODEL_PATH` in the slot env file during swaps (found in the Task 2 review).
- Tests, committed by Claude: `tests/test_deploy_script.py` and additions to `tests/test_repo_consistency.py`; see the pinned notes.

**Interfaces:**
- **`deploy-manager.sh`** takes new env overrides: `DEPLOY_PROFILES_DEST`, default `/etc/llama/models.d`; and `DEPLOY_MODELS_DIR`, default `MODELS_DIR` from the manager env, else `/opt/llama/models`.
  - The drift loop also compares each `config/models/*.toml` with `$DEPLOY_PROFILES_DEST/<same name>`, reported the same way as `.py` files.
  - Every `*.toml` in the destination with no repo counterpart is reported `EXTRA (removed on deploy)` and counts as drift.
  - Before installing, it runs `python -m manager.profiles check "$REPO/config/models" --slots "$SLOTS" --models-dir "$MODELS_DIR"` as the test user, after the suite. On failure it exits **5** with `profile validation failed; not deploying`. `SLOTS` is read from the manager env the way `HOST`/`PORT` are, defaulting to `main`.
  - Install: `install -d -m 755 "$DEPLOY_PROFILES_DEST"`, `install -m 644 -o root -g root` for each profile, then remove the extras.
- **`doctor.sh`** adds these checks, each a PASS or FAIL line like the existing ones:
  1. `/var/lib/llama-manager` exists and is owned by `_llama-mgr`.
  2. Each slot's installed unit has `EnvironmentFile=/var/lib/llama-manager/<slot>.env`, listed before its slot env file.
  3. No slot env file sets `MODEL_PATH`, `SAMPLING_ARGS`, `CTX_SIZE`, `N_GPU_LAYERS`, `MODEL_ARGS` or any `LLAMA_ARG_*`.
  4. No `*.args` files are left in `MODELS_DIR`.
  5. `manager.env` has no `SLOT_*_MODELS`.
  6. `python -m manager.profiles check /etc/llama/models.d` passes.
- **`setup.sh`** creates `/var/lib/llama-manager` (`_llama-mgr`, 755) and installs `config/models/*.toml` to `/etc/llama/models.d/`.

**Pinned by the committed tests (2026-11-01, checked against a reference implementation; thirteen hand mutants all killed).** Two points correct the Interfaces block above, which this plan got wrong; deviating from it there is correct:
- **Validation runs in both modes, before any exit decision (correction).** Run `manager.profiles check` right after the drift report and before the "Everything deployed matches" / `--check` / root-check exits, not after the suite. An invalid profile then exits 5 from `--check` too, even with no drift, and before anything installs. As planned it sat behind the root check, where nothing but a real `sudo` deploy could reach it, and `--check` (and doctor's check 7) would have called a broken profile clean.
- **Untracked profiles are refused (correction).** The dirty-tree check uses `--untracked-files=no`, so a `.toml` dropped into `config/models/` but never committed would ship unreviewed. A non-`--check` run with any `git ls-files --others --exclude-standard -- 'config/models/*.toml'` output exits 3 and names the files.
- New override `DEPLOY_PYTHON`, default `$DEST/venv/bin/python`. The validator runs as `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO" "$PY" -m manager.profiles check ...`, and its own error lines are shown, not swallowed.
- `MODELS_DIR` is `DEPLOY_MODELS_DIR`, else `MODELS_DIR` from the manager env, else `/opt/llama/models`. `SLOTS` comes from the manager env, default `main`.
- A checkout with no `config/models/*.toml` exits 5: the validator reports the missing directory.
- Profile drift lines use the file's basename in the existing `printf '  %-22s ...'` format; a missing `$DEPLOY_PROFILES_DEST` means every profile is `absent`.
- `config/manager.env`: `PROFILES_DIR` equals `ManagerConfig.from_env()`'s default and `MODEL_ENV_DIR` equals `build_slots`'s; no `SLOT_*_MODELS`, and neither `home_slot` nor `MODEL_PATH` appears anywhere in the file, comments included.
- **Not reachable from the suite** (root and host state): the install and extras-removal lines, all of `doctor.sh`'s new checks, and `setup.sh`. They are verified on the host in Task 7. Doctor check 6 runs `manager.profiles check /etc/llama/models.d --slots <the list_slots names, comma-joined> --models-dir <MODELS_DIR from the manager env, default /opt/llama/models>`. Check 2 passes when the installed unit's `EnvironmentFile=/var/lib/llama-manager/<slot>.env` line comes before the line naming the slot's env file.
- Test files in this task's test commit: `tests/test_deploy_script.py` (every clone now commits one valid profile and points all four `DEPLOY_*` paths at tmp; the two existing tests keep their assertions), additions to `tests/test_repo_consistency.py`. `tests/test_doctor.py` is unchanged.

- [ ] Steps: run the tests (they fail) → implement → full suite passes → commit `feat(ops): deploy and doctor know about profiles (Q31)`.

### Task 6: Bootstrap profiles and docs

**Files:**
- Create `config/models/Qwen3.8-27B-UD-Q6_K.toml`, `config/models/Qwen3.6-35B-A3B-UD-IQ4_NL_XL.toml` and `config/models/gemma-4-26b-a4b-it-q4_k_m.toml`.
- Update the `README.md` sections on swapping and configuration, and `manager/README.md`. The Task 2 review found stale `MODEL_PATH`/`SAMPLING_ARGS`/`CTX_SIZE`/`N_GPU_LAYERS` and regex-rewrite text around `README.md` lines 340–366, 571, 580 and 661; grep both files for those names rather than trusting the line numbers.
- Task 3 changed the API, and `README.md` still describes the old one (found in the Task 3 review; grep for `SLOT_<NAME>_MODELS`, `no_home_slot` and `GGUF`): untargeted `/swap` goes to the profile's `home`; the `trial` field; the errors `invalid_trial`, `model_not_found` (no profile, not "no GGUF"), `slot_not_allowed` and `not_verified`; a trial reloads even a loaded model; `/v1/models` lists only onboarded profiles with `home_slot`, `verified_slots` and `loaded_on`; `/status` `slots.<name>.models` comes from the profiles, plus the top-level `profile_errors`. Drop the `SLOT_<NAME>_MODELS` row from the configuration table.
- Tests: `tests/test_bootstrap_profiles.py`, committed by Claude. Each profile validates and renders the values below.

Values come from the **live** sidecars and env files, read 2026-10-30. Wherever a sidecar was silent, the value is llama.cpp b11136's default. None of the profiles has a `verified` table; `onboard` adds it.

| Profile | home | section(s) | ctx_size | n_gpu_layers | sampling (temp, top_k, top_p, min_p, presence, frequency, repeat) | flags |
|---|---|---|---|---|---|---|
| Qwen3.8-27B-UD-Q6_K | main | main | 131072 | -1 | 1.0, 20, 0.95, 0.0, 0.0, 0.0, 1.0 | `--jinja --ctx-checkpoints 4 --n-predict 32768 --spec-type draft-mtp --spec-draft-n-max 1` |
| Qwen3.6-35B-A3B-UD-IQ4_NL_XL | egpu | egpu | 131072 | -1 | 0.6, 20, 0.95, **0.05**, 0.0, 0.0, 1.0 | `--jinja --ctx-checkpoints 4 --n-predict 32768` |
| gemma-4-26b-a4b-it-q4_k_m | egpu (Q42) | egpu | 131072 | -1 | 1.0, 64, 0.95, 0.0, 0.0, 0.0, 1.0 | `--jinja --ctx-checkpoints 4 --n-predict 32768` |

Comments to carry into the profiles. Task 2 deleted them from the slot env templates and units, so take them from git: `git show 67a0f44:config/llama-server.env`, `git show 67a0f44:config/llama-server-egpu.env`, `git show 67a0f44:systemd/llama-server.service`.
- **Every profile:** the `--ctx-checkpoints 4` rationale (ggml-org/llama.cpp#23181: the default of 32 grows unbounded and can OOM the P40's VRAM, and `--cache-ram 0` doesn't stop checkpoint creation), and `--jinja` being required for tool calling.
- **Qwen3.8:** move the MTP measurement table, the KV-cache derivation and the thinking-mode sampling rationale (including the gemma `--temp 0.2` history) from `config/llama-server.env` into comments.
- **Qwen3.6:** note that `min_p` is llama.cpp's default, kept to preserve today's behaviour, and that the card says 0.0. Also carry the model-selection rationale, the KV-cache derivation, the "precise coding" sampling note, and the `n_gpu_layers = -1` auto-fit warning (experts silently moved to RAM; check `nvidia-smi` and the `common_params_fit_impl` lines) from `config/llama-server-egpu.env`.
- **gemma:** note that its sampling comes from the GGUF's `general.sampling.*` (temp 1.0, top_k 64, top_p 0.95). It has no MTP layer, so it carries no `--spec-*` flags: it's the Q42 regression case. Also note that its `ctx_size` on the P40 is unmeasured, and that `onboard`'s trial load is the measurement.

- [ ] Steps: run the tests (they fail) → write the three files and the README sections → full suite passes → commit `feat(profiles): bootstrap profiles for the three kept models (Q34, Q42)`.

### Task 7: Host cutover (the user runs these; Claude verifies each step)

Claude hands over each command once it has run its read-only half. Before starting, Claude records the `/status` output and each backend's `/proc/<pid>/cmdline`.

1. **Backups:** the two units, the two slot env files and `manager.env`, to `/root/phase3/`.
2. **Directory:** `sudo install -d -o _llama-mgr -g _llama-mgr -m 755 /var/lib/llama-manager`
3. **Deploy:** run `deploy-manager.sh`. It installs the code and profiles. The old units keep running until step 5. Nothing is verified yet, so `/swap` refuses everything that isn't a trial, which keeps swaps from landing mid-cutover.
4. **Initial model env files:** as `_llama-mgr`, run `python -m manager.profiles render /etc/llama/models.d <model> <slot> --write /var/lib/llama-manager/<slot>.env` for Qwen3.8 on main and for Qwen3.6 on egpu.
5. **New units:** install the new units and the stripped slot env files, keeping each host's real `CUDA_VISIBLE_DEVICES`. Then run `daemon-reload` and restart both llama-servers.
   - **Claude verifies** that each backend's `/proc/<pid>/cmdline` equals the slot flags followed by the rendered `MODEL_ARGS`.
6. **Onboard, then deploy:**
   - Run `scripts/onboard.py` three times: Qwen3.8 on main, Qwen3.6 on egpu, then gemma on egpu. gemma's run swaps egpu to gemma and back to Qwen3.6.
   - Claude commits the three `verified` blocks, opens the PR, the user merges, and then deploys again.
7. **Cleanup:** remove the `.args` sidecars and `SLOT_*_MODELS` from `manager.env`, then run `doctor.sh`.

**Rollback:** restore `/root/phase3/*`, run `daemon-reload`, restart the two backends, then deploy the previous merge commit.
