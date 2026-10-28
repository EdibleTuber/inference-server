TASK: list every place this file has an effect outside the Python process, or
depends on something outside it.

Include each of these: file open/read/write/rename/unlink/glob, `Path` operations
that touch the disk, subprocess or shell commands, `sudo`, `systemctl`, HTTP
requests (to any host), database calls (sqlite `execute`/`commit`/`connect`),
environment variable reads, reading GPU or system info.

One JSON object per occurrence:
{"line": N, "quote": "...",
 "kind": "file_read|file_write|fs_query|subprocess|http|db|env_read|system_info",
 "target": "path, URL, command, table, or env var name, as literally shown or 'computed'",
 "enclosing_function": "name",
 "atomic_or_partial": "for file writes only: does it write in place (open 'w') or write-temp-then-rename? else 'n/a'",
 "failure_handled_here": "yes (line of except) / no"}
