TASK: inventory error handling and background tasks in this file.

Part A — every `try:` block. One object per `except` clause, and one per
`finally`:
{"part": "A", "line": N, "quote": "the except/finally line verbatim",
 "try_line": N_of_try,
 "catches": "exception types, or 'bare' or 'Exception'",
 "handler_does": "short literal description: log / re-raise / return value / set field X / swallow (pass)",
 "enclosing_function": "name"}

Part B — every `await` call that is NOT inside any `try:` block in the same
function and sits inside a function that is a loop or runs as a background
task. List it if you are not sure:
{"part": "B", "line": N, "quote": "...", "enclosing_function": "name"}

Part C — every `asyncio.create_task`, `ensure_future`, `BackgroundTasks.add_task`,
`loop.run_in_executor`:
{"part": "C", "line": N, "quote": "...",
 "result_kept": "assigned to a variable/collection, or discarded",
 "result_awaited_or_checked": "where (line) or 'nowhere in this file'",
 "enclosing_function": "name"}
