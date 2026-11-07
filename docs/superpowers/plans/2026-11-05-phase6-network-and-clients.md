# Phase 6: Network and Clients — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The manager answers only on localhost and Tailscale, behind an optional bearer token, and agent_core is its one client library: it stops swapping on its own, sends the token, defaults to localhost, and its `/model` speaks the phase 5 swap contract. PAL and PARE move onto that agent_core.

**Architecture:**
- **agent_core first** (Tasks 1–3, released as v1.13.0). It is what PAL and PARE run, so the manager can only require a token once agent_core sends one.
- **Then the manager** (Tasks 4–6): the token check, several bound addresses in one process, and a `/status` that only reads.
- **Then the clients and the cutover** (Tasks 7–8), in the order Q16 fixes: listen three ways, move the clients one at a time, confirm from the access log, drop the LAN.

**Tech Stack:** Python 3.12.3. Manager: FastAPI 0.135.2, uvicorn 0.42.0 (httptools). agent_core 1.12.1 → 1.13.0, httpx. PAL runs agent_core 1.6.2 today, and its pyproject pins `v1.4.0`.

**Spec:** `docs/design-review-2026-10-29.md`: Q14, Q15 and Q16 in "Purpose and trust boundary", Q17 and Q18 in "Clients", and the phase 6 row of "Phases and process".

## Global Constraints

- **Three repos, one branch each:**

  | Repo | Worktree | Branch | Suite |
  |---|---|---|---|
  | inference-server | `/mnt/secondary/inference-server-wt-phase6` | `phase6/network-and-clients` | `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/ -q -p no:cacheprovider` |
  | agent_core | `/mnt/secondary/agent_core-wt-phase6` | `phase6/manager-contract` | `.venv/bin/python -m pytest -q -p no:cacheprovider`, using `/mnt/secondary/projects/agent_core/.venv` |
  | PAL | `/mnt/secondary/PAL-wt-phase6` | `phase6/agent-core-1.13` | `PYTHONPATH=/mnt/secondary/agent_core-wt-phase6 /mnt/secondary/PAL/.venv/bin/python -m pytest -q -p no:cacheprovider --continue-on-collection-errors` |

- **PAL's baseline (2026-11-05):** 628 passed and 5 collection errors (`ModuleNotFoundError: pal._legacy_tools`, in `test_chat_research_integration`, `test_client`, `test_daemon`, `test_integration` and `test_prompt_injection`). The result is the same against agent_core 1.12.1's source. Those 5 errors are out of scope; any new one is a failure.
- **Live ports:** tests never touch 8080–8084, 11434 or 2929. Tests that bind real sockets use ephemeral ports on 127.0.0.x.
- **The token never appears in logs, reprs, argv or error messages.**
- **The manager's API may only gain things.** Every response shape stays as phase 5 left it. The new 401 is the one addition.
- **Never edit a test to make it pass.** Claude owns the tests.

## How this phase runs (medium)

What changed from phase 5 (heaviest):
- **Code in the plan:** interfaces, values and invariants only. Most of this phase is wiring.
- **Tests:** Claude writes them first, checks each one fails against the pre-task code, and checks they pass against a scratch reference.
- **Review:** an ordinary diff review per task. Task 4 (the token check) is the security boundary, so it also gets an adversarial pass over request shapes walked through the middleware.
- **Mutation runs:** Tasks 1, 3, 4, 5 and 6. Task 2 is constructor wiring, Task 7 deletes code, and Task 8 is operations.
- **Releases and deploys** happen outside Qwen's sessions: the agent_core tag, PAL's and PARE's reinstall, and every step of Task 8.

## Implementer Rules (Qwen)

1. Do exactly one task, then stop and report "Done" plus a **Deviations** list.
2. Never edit a test file. If one is wrong, stop and report it.
3. Stop and report if a step's Expected result doesn't happen.
4. Never run sudo, systemctl, deploy scripts, `git push` or `git tag`, and never touch ports 8080–8084, 11434 or 2929.
5. If anything in this plan turns out to be wrong, deviating is correct; say so in Deviations.
6. Commits stage explicit paths and end with the repo's attribution lines.

## Review Focus

1. **The exemption is exactly `/health`.** `/health/`, `/HEALTH`, `//health` and `/healthz` all need the token, and so does every other path, including 404s. → Task 4, `test_only_the_exact_health_path_is_exempt`.
2. **A client without the token, once the manager requires one.** agent_core's `trigger_reindex` and `get_reindex_*` turn any non-2xx into a logged `None` (`agent_core/retrieval.py:75-119`), so a missing token would quietly stop vault reindexing rather than fail. → Task 2, `test_every_retrieval_call_sends_the_token`, plus the Task 8 order: the clients get the token before the manager requires it.
3. **Tailscale not up yet when the manager starts.** Binding an address the host doesn't have fails with `EADDRNOTAVAIL`, which would stop the manager starting, LAN and localhost included. Measured 2026-11-05: binding `100.64.0.99` fails with errno 99, and succeeds with `IP_FREEBIND`. → Task 5, `test_an_address_the_host_does_not_have_yet_still_binds`.
4. **A token read from an env file with stray whitespace** (`MANAGER_TOKEN=abc ` or an empty value) must not quietly become a different token, or no token. → Task 4, `test_an_empty_or_whitespace_token_refuses_to_start`.
5. **A `/model` swap that falls back.** The slot ends healthy on the old model, so `/model` has to report the fallback rather than a success. → Task 3, `test_a_swap_that_fell_back_is_reported_as_a_failure`.

## Decisions made while planning (beyond the spec)

- **Token transport:** `Authorization: Bearer <token>`. The manager also accepts `x-api-key: <token>`, which is the header Anthropic clients send to `/v1/messages`. opencode's OpenAI provider sends its `apiKey` as a bearer token, so opencode carries the token in `apiKey`.
- **Token storage:**
  - Manager side: `/etc/llama/manager.env` is mode 0644 (checked 2026-11-05), so the token goes in its own file, `/etc/llama/manager-token.env`, mode 0640 `root:_llama-mgr`. The unit loads it as an optional second `EnvironmentFile=-/etc/llama/manager-token.env`.
  - Client side: `<PREFIX>MANAGER_TOKEN` (`PAL_MANAGER_TOKEN`, `PARE_MANAGER_TOKEN`), set from a 0600 `EnvironmentFile` in each user unit, never as an inline `Environment=`.
- **A 401 writes no `manager.requests` line.** uvicorn's access log records it with the client address, which is what Q16 step 3 reads.
- **Several addresses in one process:** uvicorn's `--host` takes one address. `uvicorn.Server.run(sockets=...)` serves a list (`Server.startup` loops over the sockets; checked in 0.42.0), so a small entry point binds the sockets itself. `IP_FREEBIND` makes the Tailscale address independent of boot order, so there is no `After=` on the tailscale snap unit.
- **`/model` never repoints the session (owner, 2026-11-05).** A swap changes what a slot holds for every client, but repointing changes one session. The owner's real need is switching the whole server between agent work and coding, and that is a separate design: named server setups, phase 7, below. Until then, each agent sends the model its config names, and `/model` only shows what's loaded and asks for swaps. This drops agent_core's current repoint-after-a-`main`-swap.
- **The backends are already localhost-only.** From `/etc/llama/manager.env` and `ss -ltn` (2026-11-05): main's llama-server is on `127.0.0.1:8081`, egpu's on `127.0.0.1:8084`, and embeddings on `127.0.0.1:8082`. The `0.0.0.0:8080` listener is SearXNG in Docker (PAL's web search, `PAL/docs/searxng-setup.md`), and `0.0.0.0:2929` is ArcticBase. Neither belongs to this repo, and both are outside this phase.
---

## agent_core (branch `phase6/manager-contract`)

### Task 1: No automatic swap (Q17)

**Files:** `agent_core/inference.py`. Tests: `tests/test_inference.py`, plus `tests/conftest.py` if its mock `/swap` route is unused afterwards.

**The rule:** an `InferenceClient` never sends `POST /swap`. A 409 is handled like every other 4xx: on both `complete` and `stream`, the first response's `raise_for_status()` raises `httpx.HTTPStatusError` with status 409, and no retry follows. Remove `_request_model_swap`, `_is_model_not_loaded`, `_MODEL_NOT_LOADED` and the `swap_attempted` logic in `_post_with_retry` and `_stream_with_retry`. The 503 retry and the connection-drop retry are unchanged. This also removes the known bug at `inference.py:218`, where only a 200 counted as a successful swap even though the manager answers 202.

**Tests (Claude):**
- The six tests from `test_a_model_not_loaded_409_triggers_a_swap_and_one_retry` through `test_a_failed_swap_surfaces_the_original_409_not_the_swap_error`, and the streaming swap test around line 701, are replaced by:
  - `test_a_model_not_loaded_409_is_raised_without_a_swap[complete|stream]`: exactly one request, to `/v1/chat/completions`; the error's status is 409.
  - `test_no_inference_path_ever_posts_to_swap`: 409, 404, 503-then-409 and a connection drop then 409, all on a transport that fails the test on any `/swap`.
- The 503 and connection-drop retry tests stay as they are.

**Mutants:** leave the swap in the stream path only; keep the swap on 409 with a 200 or 202 answer; drop the `raise_for_status()` on 409.

### Task 2: The token, and localhost by default (Q15, Q16)

**Files:** `agent_core/config.py`, `agent_core/inference.py`, `agent_core/retrieval.py`, `agent_core/runtime.py`. Test: `tests/test_manager_token.py` (new).

**Interfaces (produces):**
- `BaseConfig.manager_token: str | None = field(default=None, repr=False)`, read from `<PREFIX>MANAGER_TOKEN`. An empty or whitespace-only value means None. The manager's own `MANAGER_TOKEN` name is not read.
- `InferenceClient(base_url, model, is_batch=False, token: str | None = None)`, with a public read-only property `token`. When the token is set, every request it sends carries `Authorization: Bearer <token>` (set on its `httpx.AsyncClient`'s default headers).
- `RetrievalClient(base_url, collection_id, token: str | None = None)`, with the same header on every call.
- The token goes only to the manager: `WebSearchClient` (SearXNG) and the URL fetcher never send it.
- `runtime.py` passes `config.manager_token` to both.
- `BaseConfig.inference_url` and `batch_inference_url` default to `"http://127.0.0.1:11434"`. `searxng_url` is unchanged.

**Tests (Claude):** committed in agent_core `tests/test_manager_token.py`; `tests/test_config.py`'s default-URL assertions now say localhost. The header tests swap a `MockTransport` into every `httpx.AsyncClient` the code builds and check the wire, so they hold whether the header is a client default or set per request.
- `test_every_inference_call_sends_the_token` (complete, stream).
- `test_a_503_retry_sends_the_token_again`.
- `test_every_retrieval_call_sends_the_token`: `search`, `get_document`, `trigger_reindex`, `get_reindex_status` and `get_reindex_job`.
- `test_no_token_sends_no_authorization_header` (complete, stream, retrieval).
- `test_the_inference_clients_token_is_readable_and_read_only`.
- `test_the_token_is_read_from_the_agents_env[pal|pare]`, with the other agent's variable set to a different value.
- `test_no_token_env_means_no_token`, with the manager's `MANAGER_TOKEN` set.
- `test_an_empty_token_env_means_no_token` (`""`, spaces, a tab).
- `test_the_token_is_not_in_the_config_repr` (`BaseConfig` and a subclass).
- `test_the_default_inference_urls_are_localhost`.
- `test_the_runtime_passes_the_token_to_both_clients`: six manager calls carry it; the SearXNG call does not.

### Task 3: `/model` speaks the phase 5 contract (Q18)

**Files:** `agent_core/commands/_builtin_impls.py` (`Model`), and `pyproject.toml` (version `1.13.0`). Tests: `tests/test_model_command.py` (new); `tests/test_builtin_commands.py` keeps the `list`, `default`, no-`/status` and unreachable-manager tests and loses the three that pinned repointing and main/batch.

**Interfaces (consumes):** `InferenceClient.token` from Task 2. Every manager call here sends it.

**Behaviour:**
- **No argument:** `GET /status`, then one line per slot, in `/status`'s order: `  <slot>: <loaded_model or "none"> (<state>)`. Then `session sends: <default_model>`. Any slot name is accepted; nothing names `main` or `batch`. If `/status` fails, fall back to the existing pointer line.
- **`list` and `default`:** unchanged, apart from the token.
- **Swap: `/model [--target <slot>] <name>`.**
  - POST `/swap` with `{"model": name}`, plus `"target"` only when given: no target means the profile's home slot. The request carries `X-Request-Id: <uuid4 hex>`. No slot names are checked on the client; the manager's 400 `invalid_target` message names the valid slots and is shown as is.
  - **200:** already loaded; report it.
  - **202:** yield `ToolProgressMessage(tool="model", arguments={"status": "swapping <slot> -> <name>"})`, with `<slot>` from the 202 body. Then poll `GET /status` every `MODEL_SWAP_POLL_SECONDS` (module constant in `_builtin_impls.py`, 2.0, read at call time) until that slot's `last_swap.id` equals the request id. A poll that fails (503, connection error) does not end the wait.
    - **Correction (Claude, 2026-11-05):** this plan first said "one interim message". It must not be a `ResponseMessage`: `DaemonConnection.command`/`command_stream` (`agent_core/client.py:104,128,152`), the CLI REPL (`agent_core/adapters/cli.py:202`) and PAL's CLI and Discord readers all stop at the first `ResponseMessage`, so the real outcome would be left in the socket and read as the next command's answer. A `StreamChunkMessage` is no better: PAL's Discord reader replaces the final text with the joined chunks (`pal/discord_interactions.py:668-671`). `ToolProgressMessage` with `arguments["status"]` is the progress convention PAL already uses.
    - Outcome `ok` or `already_loaded`: report success.
    - `fell_back` or `failed`: report the outcome and `last_swap.message` as an error.
    - The wait is capped at `MODEL_SWAP_WAIT_SECONDS` (module constant, 1500.0: two `SWAP_TIMEOUT`s of 600 plus restarts). At the cap, report that the swap is still running and that `/model` shows its state.
  - Exactly one `ResponseMessage` or `ErrorMessage` per run, and it comes last. Success and the cap are a `ResponseMessage`; `fell_back`, `failed`, a refusal and a network error are an `ErrorMessage`.
  - **Any other status:** an error with the manager's `error.message`.
  - **The session is never repointed** (see the decision above). Every outcome leaves `default_model` as it was, and the report ends with `session sends: <default_model>`.

**Tests (Claude)** (committed in agent_core; manager scripted as a `MockTransport` swapped into every `httpx.AsyncClient`; poll interval and cap patched to 0.01 s and 0.3 s). The list below is the plan's; the file also has `test_a_swap_with_target_sends_it_unchecked`, `test_a_swap_carries_a_fresh_request_id`, `test_the_slot_polled_is_the_one_the_202_names`, `test_a_202_yields_progress_not_an_interim_response`, `test_a_status_error_while_polling_does_not_end_the_wait`, `test_a_landed_swap_is_reported_as_success[ok|already_loaded]`, `test_target_without_a_name_shows_usage` and `test_no_token_sends_no_authorization_header`.
- `test_status_lists_every_slot_the_manager_reports`: three slots, `egpu` among them.
- `test_a_swap_without_target_sends_no_target`.
- `test_a_202_is_polled_until_its_own_last_swap`: an older `last_swap` id is shown first and must be ignored.
- `test_a_swap_that_fell_back_is_reported_as_a_failure`.
- `test_a_failed_swap_is_reported_as_a_failure`.
- `test_a_200_is_reported_at_once`.
- `test_no_swap_outcome_repoints_the_session[200|ok|fell_back|failed|cap]`.
- `test_a_refused_swap_shows_the_managers_message` (400 `invalid_target`).
- `test_a_swap_still_running_at_the_cap_says_so`.
- `test_every_model_call_sends_the_token`.

**Mutants:** match any `last_swap` rather than its own id; report `fell_back` as success; repoint on success; always send `target`; poll the wrong slot.

**After Task 3:** the owner tags `v1.13.0` and pushes it. Claude gives the commands; Qwen never tags.

---

## inference-server (branch `phase6/network-and-clients`)

**Task 3 review fix (2026-11-07):** agent_core `main` carried `c7a037a` (2026-10-20, after `v1.12.1`), an older `/model` 202+poll adaptation that Tasks 1–3 replace. Merge `8c4957f` records it as merged and keeps this branch's tree (its 4 tests pinned the replaced design). One fix of it is not carried: an httpx timeout stringifies to `''`, so `_builtin_impls.py`'s `"Swap failed ({name}): {exc}"` (the `/swap` POST) and `"Could not reach inference manager: {exc}"` (`list`) read blank. Name the exception type in both, keeping its text. Test: `tests/test_model_command.py::test_a_failed_call_says_what_went_wrong[blank-timeout|with-message × list|swap]` (`3ab4b82`). Add a line to the CHANGELOG 1.13.0 entry. The `v1.13.0` tag goes on this fix's commit, not `f9629aa`.
**Mutants:** the type name dropped on the `/swap` path; dropped on the `list` path; the exception text dropped (type name only) on either path.

### Task 4: The token (Q15)

**Note (Claude, from the Task 3 read):** `swap_slot`'s docstring in `manager/app.py` says "No auth; LAN-only is the trust boundary" and its untargeted branch says "agent_core clients send exactly this after a model_not_loaded 409". Both go stale in this phase: update them in this task.

**Files:** `manager/config.py`, `manager/app.py` (the new middleware and its registration), `systemd/llama-manager.service` (the second `EnvironmentFile`), `scripts/onboard.py`, `scripts/deploy-manager.sh` (its `/status` call), `README.md` (the client contract, Q18). Test: `tests/test_manager_token.py` (new), plus the scripts' existing tests.

**Interfaces:**
- `ManagerConfig.manager_token: str | None = field(default=None, repr=False)`, from `MANAGER_TOKEN`.
  - Unset: None, and no check at all.
  - Set but empty, or containing any whitespace: `ValueError` naming `MANAGER_TOKEN`, so the manager refuses to start. The message never contains the value: it goes to the journal.
  - `create_app` makes the same check on `config.manager_token` itself, at construction, not in the middleware's `__init__` (Starlette builds the middleware stack lazily, on the first request). A config built directly with `manager_token=""` would otherwise make an empty `x-api-key` match.
- `class TokenMiddleware` is pure ASGI, like `JsonOnlyMiddleware`, and outermost, so a 401 comes before a 415.
  - It checks HTTP scopes only, and only when the token is set.
  - `scope["path"] == "/health"` passes, whatever the method.
  - Every other request needs `Authorization: Bearer <token>` (the scheme matched case-insensitively) or `x-api-key: <token>`, compared with `hmac.compare_digest` on bytes.
  - Otherwise it answers 401 with `WWW-Authenticate: Bearer`. The body is `{"type": "error", "error": {"type": "authentication_error", "message": ...}}` for paths starting `/v1/messages`, and `{"error": {"type": "unauthorized", "message": ...}}` for everything else.
  - The message never echoes what the client sent.
- **`onboard.py`** sends `Authorization: Bearer $MANAGER_TOKEN` when that variable is set in its environment, as a default header on the `httpx.Client` that `main` builds. A 401 on any call exits 1 with one line saying to export `MANAGER_TOKEN`, and no traceback: today `onboard()`'s first call does `client.get("/status").json()["slots"]`, which would raise `KeyError` on a 401 body, so `main` has to catch the 401 itself (an httpx response event hook is one way). Never take the token as an argument (argv is visible in `ps`).
- **`deploy-manager.sh`** runs as root. It reads `MANAGER_TOKEN` from `/etc/llama/manager-token.env` (overridable as `DEPLOY_MGR_TOKEN_ENV` for tests) with `sed`, like its other settings, never `source`. It sends the header on its `/status` call. Its `/health` call is unchanged.
  - **The token never goes on curl's command line** (root's argv is readable by every user): pass the header on stdin, `-H @-`, or from a 0600 file, `-H @<file>`.
  - **New `--verify` mode:** the post-restart check (poll `/health` up to `HEALTH_WAIT`, then print the slot lines from `/status`) moves into a function. The deploy calls it after the restart, and `deploy-manager.sh --verify` runs it alone and exits with its result (0 healthy, 1 not), with no git, drift or root checks. That is what makes the token path testable without root, and it gives the owner a check to run in Task 8.
  - **Pre-existing bug, fix it here:** the slot dump's Python one-liner (`scripts/deploy-manager.sh:255-259` at `8d75522`) puts `\"` inside f-string braces, which Python 3.12.3 (this host's `python3`) rejects as a SyntaxError. `2>/dev/null || true` hides that, so the slot lines have never printed. Use `%` formatting or `.format()`; `test_deploy_sends_the_token_on_its_status_call` checks that a slot line prints.

**Tests (Claude):** committed in `tests/test_manager_token.py`. The list below is the plan's; the file also has `test_the_401_never_echoes_what_was_sent`, `test_health_is_exempt_whatever_the_method`, `test_the_401_uses_the_manager_shape_elsewhere`, `test_the_lifespan_still_runs_with_a_token`, `test_the_token_is_read_from_manager_token`, `test_create_app_refuses_an_empty_or_whitespace_token_too`, `test_the_unit_loads_the_token_file_optionally_after_manager_env` (exactly `/etc/llama/manager.env` then `-/etc/llama/manager-token.env`), `test_the_manager_env_template_carries_no_token`, `test_onboard_without_the_token_sends_no_header`, `test_onboard_on_a_401_exits_1_saying_to_export_the_token`, `test_onboard_takes_no_token_argument` and `test_deploy_without_a_token_file_sends_no_header`. The exemption test drives the app at the ASGI level with the exact `scope["path"]`, since an HTTP client rewrites `//health` and `/a/../b` before sending.
- `test_without_a_token_nothing_is_checked`.
- `test_a_request_without_the_token_gets_401_on_every_route`: the route list is derived from `app.routes`, not written out, plus an unknown path.
- `test_the_token_is_accepted_as_bearer_or_x_api_key`.
- `test_a_wrong_token_gets_401`, including a prefix of the right token, the right token with a trailing space, and `Basic`.
- `test_only_the_exact_health_path_is_exempt`: `/health` passes; `/health/`, `/HEALTH`, `//health` and `/healthz` get 401.
- `test_the_401_comes_before_the_json_check`.
- `test_the_401_uses_the_anthropic_shape_on_v1_messages`.
- `test_an_empty_or_whitespace_token_refuses_to_start`.
- `test_the_token_is_not_in_the_config_repr_or_any_log_line`.
- `test_onboard_sends_the_token_from_its_environment`.
- `test_deploy_sends_the_token_on_its_status_call`: through `--verify`, with a fake `curl` on `PATH` that records each call's argv and headers. The token is on the `/status` call, not on `/health`, never in any argv, and a slot line prints.

**Mutants:** `startswith("/health")`; `==` in place of `compare_digest`; check only `/v1/*`; middleware registered inside `JsonOnlyMiddleware`; empty token treated as unset; Bearer matched case-sensitively. Also, in the scripts and the rest of the middleware: strip the presented token before comparing; accept `x-api-key` only; accept Bearer only; skip the `create_app` check; put the value in the error message; let onboard's 401 reach the KeyError; deploy's header on curl's command line; deploy's token on `/health` too; the slot dump back to the f-string; the slot dump silent again on a failed `/status`; no note when the token file is unreadable. (`==` in place of `compare_digest` is expected to survive: both sides are bytes, so it is equivalent but for timing.)

**Fix round 1 (Claude's review of `a29a00c`).** One finding, from the plan, not the implementation: `--verify` needs no root, but the token file is 0640 `root:_llama-mgr`, so run unprivileged it reads no token, `/status` answers 401, and the slot dump fails silently behind `2>/dev/null || true`. That is the same silence that hid the f-string bug. Tests `test_deploy_verify_says_when_it_cannot_read_the_token_file` and `test_deploy_verify_says_when_the_slot_dump_fails[401|empty|garbage]` fail on `a29a00c`. Required lines, exactly:
- when the token file exists and is not readable, after the healthy line: `  note: <token file path> is not readable; run with sudo for the slot dump`
- when the dump prints no slots, for any reason: `  no slot dump: /status did not answer with slots`

The health verdict and exit code do not change.

### Task 5: Several addresses, one process (Q14, Q16)

**Files:** `manager/config.py`, `manager/serve.py` (new), `systemd/llama-manager.service` (`ExecStart`), `config/manager.env` (`LISTEN_HOSTS`), `scripts/onboard.py` and `scripts/deploy-manager.sh` (the base URL), `README.md`. Test: `tests/test_listen.py` (new).

**Interfaces:**
- `ManagerConfig.listen_hosts: tuple[str, ...]`, from `LISTEN_HOSTS` (comma separated, whitespace around items stripped).
  - When unset, it is `(HOST,)`, so an env file without the setting behaves as today.
  - Each item must parse with `ipaddress.ip_address`. No duplicates. A wildcard (`0.0.0.0` or `::`) only alone.
  - Any violation raises `ValueError` naming `LISTEN_HOSTS`.
- `manager/serve.py`:
  - `bind_sockets(hosts: Sequence[str], port: int) -> list[socket.socket]`: one listening TCP socket per host, with `SO_REUSEADDR` and `IP_FREEBIND` (`IPV6_FREEBIND` for v6), set non-blocking.
  - `main() -> None`: loads the config, binds, and runs `uvicorn.Server(uvicorn.Config("manager.app:create_app", factory=True, timeout_keep_alive=120, log_level="info")).run(sockets=...)`. It keeps uvicorn's default `http="auto"`; phase 5's client-gone check relies on what auto picks.
  - A bind failure exits non-zero, naming the host and the error.
- The unit's `ExecStart` becomes `/opt/llama/manager/venv/bin/python -m manager.serve`. The template sets `LISTEN_HOSTS=127.0.0.1,192.168.1.14,100.82.222.92` with a comment line above it (no trailing comment; see phase 5's branch review).
- The scripts' base URL comes from the first loopback address in `LISTEN_HOSTS`. With none, the rule is the old one: a wildcard means 127.0.0.1, otherwise the first host. With `LISTEN_HOSTS` unset, the old `HOST` rule applies.

**Corrections from writing the tests (Claude, 2026-11-06).** These override the interface lines above where they differ:
- **`IP_FREEBIND` has no constant in this host's Python (3.12.3).** `socket.IP_FREEBIND` and `socket.IPV6_FREEBIND` raise AttributeError. Use `getattr(socket, "IP_FREEBIND", 15)` and `getattr(socket, "IPV6_FREEBIND", 78)` (the Linux values).
- **`bind_sockets`:**
  - picks the family from the address (`AF_INET6` for a v6 address);
  - listens, then sets non-blocking;
  - on a failed bind, closes every socket it already bound, then raises `OSError` whose message contains the host;
  - `main` catches it and exits non-zero with a message containing the host and the error's `strerror` (for example `Address already in use`).
- **`main`:** reference `uvicorn.Server` and `uvicorn.Config` as attributes of `import uvicorn` (the tests replace `uvicorn.Server`). The tests run `main` in-process: a subprocess that a mutant let start would run `create_app`'s lifespan against the live slot ports, outside conftest's guard.
- **`listen_hosts`:**
  - the field defaults to `()` for configs built directly; `from_env` always fills it;
  - set but empty or blank is refused like an empty item;
  - duplicates are compared as addresses, so `::1` and `0:0:0:0:0:0:0:1` are the same;
  - when `LISTEN_HOSTS` is unset, `(HOST,)` gets the same checks, and the error names `HOST`.
- **Scripts' base URL:**
  - any `127.0.0.0/8` address and `::1` count as loopback (`ip_address(...).is_loopback`);
  - a v6 address is bracketed (`http://[::1]:PORT`);
  - a lone `::` means `[::1]`, a lone `0.0.0.0` means `127.0.0.1`;
  - the deploy script reads `LISTEN_HOSTS` with `sed`, like its other settings.
- **Template:** keep `HOST`; its comment says that `LISTEN_HOSTS`, when set, replaces it.

**Tests (Claude)** (`tests/test_listen.py`; before the fix, all fail except 4 cases of `LISTEN_HOSTS` unset, which guard today's behaviour):
- `test_listen_hosts_default_to_host`, `test_listen_hosts_are_parsed_and_stripped`, `test_listen_hosts_take_ipv6_and_a_lone_wildcard`.
- `test_a_bad_listen_hosts_names_the_key[not-an-ip|duplicate|duplicate-spelled-differently|wildcard-with-another|v6-wildcard-with-another|two-wildcards|empty-item|empty|blank]`, `test_a_bad_host_without_listen_hosts_names_host`.
- `test_each_socket_is_bound_listening_nonblocking_and_freebind`: v4 and v6, and the FREEBIND option read back from the socket.
- `test_an_address_the_host_does_not_have_yet_still_binds[100.64.0.99|fd00::99]`: first checks that a plain socket cannot bind the address, so the test can't pass vacuously on a host with `ip_nonlocal_bind`.
- `test_a_port_already_taken_raises_naming_the_host`, which also checks that nothing bound before the failure is leaked.
- `test_one_server_answers_on_every_bound_address`: real uvicorn on 127.0.0.1, 127.0.0.2 and ::1, same port. A shared counter proves one process answers all three.
- `test_main_runs_one_uvicorn_on_every_listen_host`: config `app`, `factory`, `timeout_keep_alive`, `log_level` and `http == "auto"`. `test_main_exits_nonzero_naming_the_host_when_a_bind_fails`.
- `test_the_unit_runs_manager_serve`, `test_the_template_listens_on_loopback_lan_and_tailnet`.
- `test_onboard_uses_a_loopback_listen_host[...]` and `test_deploy_uses_a_loopback_listen_host[...]`: the same nine cases for both scripts.

**Mutants:** drop `IP_FREEBIND`; bind only the first host; allow a wildcard alongside other hosts; the scripts take the first host rather than the loopback one. Also: drop `IPV6_FREEBIND`; leave a socket blocking; leak the sockets bound before a failure; compare duplicates as strings; accept an empty `LISTEN_HOSTS` as unset; skip the checks on the `HOST` fallback; `http="h11"` in `main`; no brackets on a v6 URL; `::` mapped to `127.0.0.1`; `main` swallows the bind error and exits 0.

### Task 6: `/status` only reads (design feedback, 2026-11-05)

**Why:** `GET /status` asks every idle, unhealthy or failed slot's consumer to probe before it answers (`manager/app.py:1536`, `reprobe_all_slots_concurrently`). Every probe the consumer runs is judged for the supervisor (`_serve_probe_requests` → `_apply_supervision`). So reading `/status` has side effects:
- a client polling it moves supervisor restarts earlier (phase 5's branch review, finding 4);
- each poll puts load on the backends;
- Task 3's `/model` polls `/status` every 2 s during a swap, which would probe the *other* slot every 2 s.

The reason for the probe (`app.py:1528-1535`) was that a backend ready after startup showed as down for the life of the process. Since phase 5, the supervisor probes every idle, unhealthy or failed slot every `SUPERVISOR_PROBE_INTERVAL` seconds, so that case is covered without it.

**Files:** `manager/app.py` (the `/status` handler, and `ServerState.reprobe_all_slots_concurrently` if nothing else calls it), `manager/slots.py`, `README.md` (the `/status` row, and the supervisor paragraph's "whoever asked for it" sentence). Test: `tests/test_status_read_only.py` (new). Existing tests that pin the probe-on-read are rewritten in Claude's commit, which names each one.

**Interfaces:**
- `SlotState.last_probe_at: float | None`, in `time.monotonic()`. Stamped by `SlotState.probe` when its backend call returns or fails, on every path, whoever asked. None until the first probe.
- `/status` gains one key per slot, `last_probe_age_seconds`: the age of `last_probe_at`, rounded to 1 decimal like the other ages, or null. Every other key is unchanged.
- `GET /status` asks for no probe and waits for nothing.
- Unchanged: the routing path's `refresh` (an unhealthy slot with the requested model), the chat 409 path's `reprobe_all_slots`, and the startup refresh.

**Notes from writing the tests (Claude, 2026-11-06):**
- **Stamp after the call.** Stamp `last_probe_at` with `time.monotonic()` read after `client.get` returns or raises, as `time.monotonic()` from `slots.py`'s own `import time`; the stamp test replaces `manager.slots.time`. Every return path of `probe` must have stamped by the time it returns: the exception path, non-200 (503 included), non-JSON, malformed and healthy.
- **Remove `ServerState.reprobe_all_slots_concurrently`.** `/status` was its only caller. Leave `reprobe_all_slots` (the chat 409 path) and `refresh` as they are.
- **Fix the comments.** The comment block above the `/status` handler and `_serve_probe_requests`'s docstring ("startup, /status and routing ask for a probe") both go stale and need updating.

**Tests (Claude)** (`tests/test_status_read_only.py`, reusing `tests/test_state_machine.py`'s fixtures). Before the fix, all fail except the last, which guards behaviour already correct:
- `test_status_asks_for_no_probe[idle|unhealthy|failed]`: no backend probe, no `probe` call, nothing pending, and the slot shows what it already held.
- `test_status_waits_for_no_consumer`: answers in under 1 s with no consumer running (it used to wait out `REFRESH_TIMEOUT`).
- `test_polling_status_does_not_move_the_supervisor_count`: `supervisor_failures + 2` polls with egpu answering 500. The failure count, `restarts_suppressed` and `restarts` stay 0.
- `test_status_reports_the_age_of_the_last_probe`: null before any probe. A number after one. Rounded to 1 decimal. The key set is exactly the old keys plus `last_probe_age_seconds`.
- `test_every_probe_path_stamps_the_probe_time[healthy|loading|failed|raised|malformed]`: the clock moves during the call, and the stamp must be the later value.
- `test_a_backend_that_comes_up_later_is_seen_by_the_supervisor` (guard).

**Removed in Claude's commit,** because they pin the probe-on-read:
- `tests/test_endpoints.py::test_status_reprobes_before_reporting`;
- `tests/test_endpoints.py::test_status_survives_a_raising_slot_probe`, with its now-unused `_FakeV1ModelsServer` and `fake_v1_models_backend`;
- `tests/test_state_machine.py::test_status_probes_an_idle_or_unhealthy_slot_through_its_consumer`.

The module docstring of `test_state_machine.py` now says `/status` asks for no probe.

**Mutants:** keep the probe for unhealthy slots only; stamp only healthy probes; stamp before the call. Also: stamp only on the exception path; skip the stamp on a non-JSON or malformed 200; report the age unrounded; report `last_probe_at` itself instead of its age; `/status` still awaits `refresh` on a slot with a failed state.

---

## PAL and PARE (branch `phase6/agent-core-1.13` in PAL; PARE needs only its pin)

### Task 7: PAL moves to agent_core 1.13 (Q18)

**Files:**
- Delete PAL's own `/model`: the `Model` command in `pal/commands/domain.py`, and `_get_manager_status`, `_model_status_text`, `_dispatch_model_command` and `_request_model_swap` in `pal/agent.py`, with their tests in `tests/test_model_command.py`. agent_core's builtin `/model` takes over.
- `pal/agent.py:205` (the batch `InferenceClient`) and `pal/backfill_main.py:55` pass `token=self.config.manager_token` (or `config.manager_token`).
- `pyproject.toml` pins `agent_core @ git+https://github.com/EdibleTuber/agent_core.git@v1.13.0`.
- `README.md`: `PAL_INFERENCE_URL` now defaults to localhost; add `PAL_MANAGER_TOKEN`.

**Tests (Claude):**
- `test_pal_registers_agent_cores_model_command`: the registry's `model` command is agent_core's `Model`.
- `test_every_pal_inference_client_gets_the_token`.
- PAL's `tests/test_config.py:14,40,92` pin the old `192.168.1.14` defaults and fail against agent_core 1.13 (measured 2026-11-05: 3 failed, 625 passed, the 5 baseline collection errors). Claude updates them with these tests, comparing against `BaseConfig()` rather than a new literal.
- Expected suite result: the baseline above, minus the deleted tests, plus these.

**Notes from writing the tests (Claude, 2026-11-07):**
- Work in the PAL worktree `/mnt/secondary/PAL-wt-phase6` (branch `phase6/agent-core-1.13`). It has no venv of its own. Run the suite with PAL's venv against the worktree and agent_core 1.13:
  `PYTHONPATH=/mnt/secondary/PAL-wt-phase6:/mnt/secondary/agent_core-wt-phase6 /mnt/secondary/PAL/.venv/bin/python -m pytest -q -p no:cacheprovider --continue-on-collection-errors`
  The 5 collection errors (`test_chat_research_integration`, `test_client`, `test_daemon`, `test_integration`, `test_prompt_injection`: modules removed long ago) are the baseline and make the exit status 1. Judge by the summary line: no failures, and errors in those 5 files only.
- **A third manager client the plan missed:** `scripts/retrieval_eval/run.py:61` builds a `RetrievalClient` with no token, and the manager will refuse it with 401. It reads `PAL_INFERENCE_URL` from the environment already. Pass `token=os.environ.get("PAL_MANAGER_TOKEN") or None` the same way.
- Delete the whole `PALModel` class from `pal/commands/domain.py`, its export from `pal/commands/__init__.py` (import and `__all__`), its import and `commands` entry in `pal/agent.py`, and the four helpers. Update the comment above `PALAgent.commands` that lists "model" among the overridden builtins. Drop any import left unused.
- Tests (Claude's commit), in `tests/test_agent_core_1_13.py`:
  - `test_pal_registers_agent_cores_model_command`;
  - `test_pals_own_model_helpers_are_gone[4 names]`;
  - `test_pal_commands_no_longer_exports_its_model_command`;
  - `test_the_batch_client_gets_the_token[same-manager|own-url]`;
  - `test_the_backfill_client_gets_the_token`;
  - `test_every_pal_manager_client_is_given_the_token`: an AST scan of `pal/` and `scripts/` for every `InferenceClient(...)` or `RetrievalClient(...)` call without a `token=` keyword.
  Also: `tests/test_config.py` compares against `BaseConfig()` and gains `test_load_config_reads_the_manager_token` (a guard; it passes before the fix). `tests/test_commands.py` and `tests/test_commands_drift.py` drop `PALModel`. `tests/conftest.py`'s `make_pal_agent` passes the token like 1.13's `run_daemon`. `tests/test_model_command.py` is deleted.
- Measured before the fix: exactly the new tests fail (10), 620 pass, the 5 baseline errors. With a reference fix: no failures twice, the 5 baseline errors.
- **Mutants:**
  - `PALModel` kept in `PALAgent.commands`;
  - the batch client built without the token;
  - the backfill client built without the token;
  - the retrieval_eval client built without the token;
  - the batch client given `token=config.manager_token` only when `config.batch_inference_url == config.inference_url`.

**PARE:** pin bump to `v1.13.0` only. PARE sets no inference URL in its unit (it uses the default), so 1.13's localhost default moves it. That is the "new PARE inference-URL env" of Q16, confirmed as `PARE_INFERENCE_URL` (`pare/config.py:51-53`, `agent_name="pare"`). The pin bump is part of Task 8.

### Task 8: Cutover (owner; Claude gives each command and checks each result)

Each step is checked before the next one:

1. **Three-way listen, no token yet.** Deploy the manager (Tasks 4–6): `sudo ./scripts/deploy-manager.sh` from the merged main, plus the unit install and `daemon-reload` that deploy doesn't do. Add `LISTEN_HOSTS=127.0.0.1,192.168.1.14,100.82.222.92` to `/etc/llama/manager.env`. **Check:** `ss -ltn` shows all three `:11434`.
2. **agent_core v1.13.0 into PAL and PARE.** Reinstall each venv from its pin, restart `pal-daemon` and `pare-daemon`, and change `PAL_INFERENCE_URL` to `http://127.0.0.1:11434` (or remove it). opencode: `baseURL` to `http://127.0.0.1:11434/v1`. **Check:** each client's next request shows client `127.0.0.1` in the manager's access log.
3. **The token.**
   - Generate it with `python -c 'import secrets; print(secrets.token_urlsafe(32))'`, never in shell history or argv: write it straight into the files.
   - Write it first to the clients' 0600 env files and opencode's `apiKey`, and restart the clients.
   - Then write `/etc/llama/manager-token.env` (0640 `root:_llama-mgr`) and restart the manager.
   - Write the line as `MANAGER_TOKEN=<token>` with no quotes: systemd strips quotes, but the deploy script reads the file with `sed` and would send them, so its `/status` call would get 401.
   - **Check:** `/status` without the header gets 401, and with it gets 200; one chat each from PAL, PARE and opencode succeeds.
4. **Confirm nothing uses the LAN.** Watch the access log for a full day of normal use: `journalctl -u llama-manager -o cat | grep -E '^INFO: +192\.168\.1\.'` must be empty after step 2's restarts. Then drop `192.168.1.14` from `LISTEN_HOSTS` and restart. **Check:** `ss -ltn` shows only `127.0.0.1:11434` and `100.82.222.92:11434`.

---

## After this phase (owner, 2026-11-05)

- **Phase 7, named server setups.** One request switches the whole server between named setups, such as "agents" (Gemma 4 on `egpu`) and "coding" (Qwen 3.6 on `egpu`). The setups would be a file in the repo next to the profiles, and the request would queue one swap per slot through phase 5's queue. The agents show what's loaded and an operator can ask for a switch; the agents never switch on their own. Open questions for its design:
  - which slot each agent uses in each setup;
  - how an agent learns the model its setup gives it, instead of a fixed `PAL_MODEL`.
- **Phase 7 must also decide** (from Claude's design feedback, 2026-11-05):
  - **Who owns a slot.** Profiles pin each model to one slot, clients name models, and swaps are server-wide. A setup should say which agent uses which slot, not just which model each slot holds.
  - **What a client sees during a switch.** Today a request for the outgoing model gets 503 `slot_unavailable`, and one for the incoming model gets 409. agent_core's 503 retry (5 tries at `Retry-After: 5`) gives up long before a cold swap ends.
- **Before phase 7: move the consumer out of `app.py`.** `app.py` is 1921 lines, and `_process_request_item` alone is 238. Most phase 5 findings were in or next to it.
- **For the agent audits: retrieval shares the manager's process.** The `/collections` routes, the vector DB and embeddings run with inference, so a manager restart takes PAL's memory down too, and one token guards both.
- **Code audits of PAL, PARE and agent_core**, after the inference-server audit (this design review) is finished. The Qwen models' thinking handling inside the agents is one known item; it's why Gemma 4 currently does better there.
