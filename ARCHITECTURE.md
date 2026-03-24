# llama-cli Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        macOS Host (Apple Silicon)                    │
│                                                                     │
│  ┌──────────────┐     HTTPS (TLS)    ┌────────────────────────┐     │
│  │              │  ──────────────►   │    Auth Proxy           │     │
│  │   CLI (TUI)  │  JWT Bearer Token  │    (Docker Container)   │     │
│  │              │  ◄──────────────   │    port 1337            │     │
│  │  python REPL │     SSE Stream     │                        │     │
│  └──────┬───────┘                    │  ┌──────────────────┐  │     │
│         │                            │  │ JWT Validation    │  │     │
│         │ Keychain                   │  │ Model Routing     │  │     │
│         │ (JWT, TLS, DB keys)        │  │ Alias Resolution  │  │     │
│         │                            │  └────────┬─────────┘  │     │
│         │                            └───────────┼────────────┘     │
│         │                                        │                  │
│         │                          ┌─────────────┴──────────────┐   │
│         │                          │ host.docker.internal       │   │
│         │                          │                            │   │
│         │              ┌───────────┴───┐      ┌────────────────┴┐  │
│         │              │  llama.cpp    │      │  MLX Server     │  │
│         │              │  port 8081    │      │  port 8082      │  │
│         │              │  user: _llama │      │  user: _mlx     │  │
│         │              │  Metal GPU    │      │  Metal GPU      │  │
│         │              └───────┬───────┘      └────────┬────────┘  │
│         │                      │                       │           │
│         │              ┌───────┴───────┐      ┌────────┴────────┐  │
│         │              │ /Users/Shared │      │ /Users/Shared   │  │
│         │              │ /llama/models │      │ /llama/mlx-     │  │
│         │              │ (GGUF files)  │      │  models/ + venv │  │
│         ▼              └───────────────┘      └─────────────────┘  │
│  ┌──────────────┐                                                  │
│  │ ~/.local/    │                                                  │
│  │ share/       │                                                  │
│  │ llama-cli/   │                                                  │
│  │ history.db   │                                                  │
│  └──────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

All services bind exclusively to `127.0.0.1`. Nothing is exposed to the network.

---

## Components

### CLI (`cli/python-cli/`)

Full-screen terminal application built with `prompt_toolkit`. Manages project context, conversation history, and authentication.

```
┌───────────────────────────────────────────┐
│ main.py    ─── REPL loop, slash commands  │
│ terminal.py ── full-screen TUI layout     │
│ render.py  ─── streaming token display    │
│ client.py  ─── HTTPS + SSE to proxy      │
│ auth.py    ─── JWT from Keychain          │
│ config.py  ─── YAML + env var loading     │
│ context.py ─── file scanning + budgeting  │
│ session.py ─── SQLite persistence         │
│ lexer.py   ─── syntax highlighting        │
└───────────────────────────────────────────┘
```

### Auth Proxy (`proxy/`, Docker)

FastAPI application that validates JWT tokens, routes requests to the correct backend, and translates model aliases.

```
Request flow:

  POST /v1/chat/completions
    │
    ▼
  auth.py ── validate JWT (HS256)
    │          reject → 401
    ▼
  router.py ── extract model from body
    │           look up model_routes[model]
    │           rewrite model field via model_aliases
    ▼
  httpx ── forward to backend (8081 or 8082)
    │
    ▼
  SSE passthrough back to CLI
```

**Routing table:**

| Alias               | Backend                            | Port |
|----------------------|------------------------------------|------|
| `qwen-32b`          | `http://host.docker.internal:8081` | 8081 |
| `qwen-32b-mlx`      | `http://host.docker.internal:8082` | 8082 |
| `qwen-32b-mlx-8bit` | `http://host.docker.internal:8082` | 8082 |

**Alias resolution** (MLX backends require the full model path in requests):

| Alias               | Rewritten to                                                     |
|----------------------|------------------------------------------------------------------|
| `qwen-32b-mlx`      | `/Users/Shared/llama/mlx-models/Qwen2.5-Coder-32B-Instruct-4bit` |
| `qwen-32b-mlx-8bit` | `/Users/Shared/llama/mlx-models/Qwen2.5-Coder-32B-Instruct-8bit` |

### Backend Servers (launchd)

Both run as unprivileged system users managed by launchd. `KeepAlive` ensures auto-restart on crash; `ThrottleInterval` prevents restart loops.

```
/Library/LaunchDaemons/
  ├── com.llama-cli.server.plist       → _llama → llama-server (8081)
  └── com.llama-cli.mlx-server.plist   → _mlx   → mlx_lm.server (8082)
```

---

## Security Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│ TRUST BOUNDARY: macOS Host                                      │
│                                                                 │
│  ┌─────────────────────────┐                                    │
│  │ User Space              │                                    │
│  │                         │                                    │
│  │  CLI process            │                                    │
│  │  ├─ Reads Keychain ────────────────────────────────┐         │
│  │  ├─ Writes ~/.local/share/llama-cli/history.db     │         │
│  │  └─ HTTPS to localhost:1337                        │         │
│  └─────────────────────────┘                          │         │
│                                                       │         │
│  ┌─────────────────────────────────────────┐          │         │
│  │ TRUST BOUNDARY: Docker Container        │          │         │
│  │                                         │          │         │
│  │  Auth Proxy (unprivileged)              │          │         │
│  │  ├─ read-only root filesystem           │          │         │
│  │  ├─ all capabilities dropped            │          │         │
│  │  ├─ secrets mounted read-only ◄─────────┘         │         │
│  │  │   /certs/localhost.crt (ro)                     │         │
│  │  │   /certs/localhost.key (ro)                     │         │
│  │  │   /run/secrets/jwt-secret (ro)                  │         │
│  │  └─ HTTP to host backends (no TLS)                 │         │
│  └─────────────────────────────────────────┘          │         │
│                                                       │         │
│  ┌──────────────────────┐  ┌──────────────────────┐   │         │
│  │ TRUST BOUNDARY:      │  │ TRUST BOUNDARY:      │   │         │
│  │ _llama user          │  │ _mlx user            │   │         │
│  │                      │  │                      │   │         │
│  │  llama-server        │  │  mlx_lm.server       │   │         │
│  │  ├─ no shell         │  │  ├─ no shell         │   │         │
│  │  ├─ no home dir      │  │  ├─ no home dir      │   │         │
│  │  ├─ Metal GPU only   │  │  ├─ Metal GPU only   │   │         │
│  │  └─ read-only models │  │  └─ own models dir   │   │         │
│  └──────────────────────┘  └──────────────────────┘   │         │
│                                                       │         │
│  ┌──────────────────────────────────────────┐         │         │
│  │ macOS Keychain (Secure Enclave)          │ ◄───────┘         │
│  │                                          │                   │
│  │  com.llama-cli.jwt-secret  (32B base64)  │                   │
│  │  com.llama-cli.tls-key     (EC PEM)      │                   │
│  │  com.llama-cli.db-key      (32B base64)  │                   │
│  └──────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘

NETWORK BOUNDARY: All ports bound to 127.0.0.1 only
  ├── :1337  Auth Proxy (HTTPS, JWT required)
  ├── :8081  llama.cpp (HTTP, no auth — internal only)
  └── :8082  MLX server (HTTP, no auth — internal only)
```

### Security Properties

| Layer              | Mechanism                                                         |
|--------------------|-------------------------------------------------------------------|
| **Transport**      | TLS 1.2+ with EC P-256 certificate; CLI pins cert                 |
| **Authentication** | JWT HS256, 1-hour TTL, auto-refresh 60s before expiry             |
| **Secrets**        | macOS Keychain (hardware-backed); never persisted to disk         |
| **Process**        | Dedicated system users (`_llama`, `_mlx`) with no shell/home     |
| **Container**      | Read-only root, all caps dropped, secrets RO-mounted, tmpfs /tmp  |
| **Network**        | Loopback-only binding on all ports; zero external exposure        |
| **Database**       | Parameterized queries; optional SQLCipher encryption              |
| **UID allocation** | Dynamic scan of 300–350 range avoids macOS system ID collisions   |

### Threat Model

| Threat                         | Mitigation                                              |
|--------------------------------|---------------------------------------------------------|
| Network eavesdropping          | TLS on proxy; all ports loopback-only                   |
| Unauthorized inference         | JWT required on every request                           |
| Secret exfiltration            | Keychain-only storage; disk secrets are ephemeral       |
| Backend compromise             | System user isolation; no shell; read-only model dirs   |
| Container escape               | Read-only root; no capabilities; unprivileged user      |
| SQL injection                  | Parameterized queries throughout                        |
| Crash → downtime               | `KeepAlive` with `ThrottleInterval` auto-restart        |
| UID/GID collision              | Dynamic allocation scanning both UIDs and GIDs          |
| GPU memory exhaustion (OOM)    | `--prompt-cache-size` limit; `KeepAlive` auto-restart   |

---

## Data Flow

### Request Lifecycle

```
 User types prompt
       │
       ▼
 CLI: context.py assembles payload
       │  ┌────────────────────────────────┐
       │  │ system prompt + {context}      │
       │  │ project files (within budget)  │
       │  │ conversation history (sliding) │
       │  │ user message                   │
       │  └────────────────────────────────┘
       │
       ▼
 CLI: auth.py generates/refreshes JWT
       │
       ▼
 HTTPS POST https://localhost:1337/v1/chat/completions
   Headers: Authorization: Bearer <JWT>
   Body:    { model: "qwen-32b-mlx", messages: [...], stream: true }
       │
       ▼
 Proxy: JWT validation (HS256)
       │
       ▼
 Proxy: route lookup → http://host.docker.internal:8082
 Proxy: alias rewrite → model field becomes full path
       │
       ▼
 HTTP POST to MLX server (or llama.cpp)
       │
       ▼
 SSE stream: data: {"choices":[{"delta":{"content":"..."}}]}
       │
       ▼
 Proxy: passthrough stream to CLI
       │
       ▼
 CLI: render.py streams tokens to terminal
 CLI: session.py persists message to SQLite
```

### Token Budget Allocation

```
 Context Window: 32,768 tokens
 ┌──────────────────────────────────────────────────┐
 │ Response Headroom          │ 4,000 tokens        │
 ├────────────────────────────┼─────────────────────┤
 │ Project File Context       │ ≤ 16,000 tokens     │
 │  (scanned files within     │                     │
 │   budget, exclude patterns │                     │
 │   applied, ~4 chars/token) │                     │
 ├────────────────────────────┼─────────────────────┤
 │ Conversation History       │ remaining tokens    │
 │  (sliding window: keeps    │                     │
 │   first msg + last msg,    │                     │
 │   fills middle with recent)│                     │
 ├────────────────────────────┼─────────────────────┤
 │ System Prompt              │ (within history)    │
 └──────────────────────────────────────────────────┘
```

---

## File System Layout

```
/Users/Shared/llama/
  ├── models/                          owner: _llama  mode: 555
  │   └── qwen2.5-coder-32b-instruct-q4_k_m.gguf
  ├── mlx-models/                      owner: _mlx    mode: 755
  │   ├── Qwen2.5-Coder-32B-Instruct-4bit/
  │   └── Qwen2.5-Coder-32B-Instruct-8bit/
  └── mlx-venv/                        owner: _mlx
      └── bin/python → mlx_lm==0.31.1

~/.config/llama-cli/
  ├── config.yaml                      user config
  ├── localhost.crt                    TLS public cert
  └── system-prompt.txt               optional custom prompt

~/.local/share/llama-cli/
  └── history.db                       SQLite (sessions, messages, context)

/usr/local/var/log/
  ├── llama-server.log                 owner: _llama
  ├── llama-server.err                 owner: _llama
  ├── mlx-server.log                   owner: _mlx
  └── mlx-server.err                   owner: _mlx

/Library/LaunchDaemons/
  ├── com.llama-cli.server.plist       owner: root:wheel
  └── com.llama-cli.mlx-server.plist   owner: root:wheel
```

---

## Startup Sequence

```
1. Initial setup (once):     sudo scripts/setup.sh
                              ├─ create _llama, _mlx users
                              ├─ create model dirs
                              ├─ generate secrets → Keychain
                              ├─ generate TLS cert
                              ├─ install launchd plists
                              └─ create config.yaml

2. Start services:           scripts/start.sh
                              ├─ launchctl load llama.cpp plist
                              ├─ extract secrets from Keychain → tmpfiles
                              ├─ docker compose up -d --build
                              ├─ wait for proxy health check
                              └─ delete tmpfiles

3. (Optional) Swap model:    sudo scripts/swap-model.sh <alias>
                              ├─ stop current backend
                              ├─ sync plist from repo
                              ├─ update model path via PlistBuddy
                              ├─ launchctl load
                              └─ wait for health check

4. Run CLI:                  llama-cli --project /path/to/code
```

---

## Database Schema

```sql
sessions
  ├── id           TEXT PRIMARY KEY     -- UUID
  ├── name         TEXT                 -- optional display name
  ├── project      TEXT NOT NULL        -- project path
  ├── model        TEXT NOT NULL        -- model alias
  ├── created_at   TIMESTAMP
  ├── updated_at   TIMESTAMP
  ├── context_set  TEXT NOT NULL        -- JSON array of file paths
  └── token_budget INTEGER DEFAULT 16000

messages
  ├── id           INTEGER PRIMARY KEY AUTOINCREMENT
  ├── session_id   TEXT → sessions(id)
  ├── role         TEXT                 -- user | assistant | system
  ├── content      TEXT
  ├── tokens       INTEGER
  ├── partial      BOOLEAN DEFAULT 0   -- interrupted response flag
  └── created_at   TIMESTAMP

context_files
  ├── session_id   TEXT → sessions(id)  ┐
  ├── file_path    TEXT                 ├── composite PK
  ├── content      TEXT
  ├── tokens       INTEGER
  └── loaded_at    TIMESTAMP
```

---

## Models

| Alias               | Architecture | Quant | Backend   | Speed    | Storage |
|----------------------|-------------|-------|-----------|----------|---------|
| `qwen-32b`          | Qwen2.5-Coder-32B | Q4_K_M | llama.cpp | ~17 t/s | ~20 GB |
| `qwen-32b-mlx`      | Qwen2.5-Coder-32B | 4-bit  | MLX       | ~17 t/s | ~18 GB |
| `qwen-32b-mlx-8bit` | Qwen2.5-Coder-32B | 8-bit  | MLX       | ~4.3 t/s | ~31 GB |
