TASK: list every write to shared mutable state in this file.

Shared mutable state means: attributes on objects that outlive one request
(e.g. anything reached through `self.`, `slot.`, `server.`, `app.state`),
module-level variables, dict/list items mutated in place (`x["k"] = ...`,
`.append`, `.pop`, `.clear`, `.set()`, `.drain()`), and counters (`+=`, `-=`).
Ordinary local variables are excluded.

One JSON object per write:
{"line": N, "quote": "...",
 "target": "e.g. slot.healthy, item['error'], self._depth",
 "new_value": "literal or expression",
 "enclosing_function": "name",
 "locks_held": "names of locks held by an enclosing `async with` in this function at this line, or 'none'",
 "awaits_between_read_and_write": "if the same function reads this target earlier and an `await` occurs between that read and this write, give the await line; else 'none'"}
