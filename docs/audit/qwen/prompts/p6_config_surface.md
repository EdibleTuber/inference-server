TASK: build a cross-reference of every configuration key across the files
provided (Python config loaders, env files, systemd units, shell scripts).
Each file is introduced by a line `===== FILE: <path> =====`.

A configuration key is: an environment variable name read in Python
(`os.getenv`, `os.environ`, `_int_env`, or a computed `f"SLOT_{...}_X"` pattern),
a `KEY=value` line in an env file, a `${KEY}` or `$KEY` reference in a systemd
unit or script, or an `Environment=` / `EnvironmentFile=` line.

One JSON object per (key, file, line) occurrence:
{"key": "NAME (write computed names with the pattern, e.g. SLOT_<NAME>_PORT)",
 "file": "path as given in the FILE header", "line": N, "quote": "...",
 "role": "read_by_code|defined|referenced|default_value",
 "value": "literal value or default shown, or 'unknown'"}

After all occurrences, output one final object per key:
{"summary_key": "NAME", "read_by_code": true/false, "defined_in": ["files"],
 "referenced_in": ["files"], "defaults_differ": "list of differing default/literal values, or 'no'"}
