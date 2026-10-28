TASK: list every place this file bounds or waits on time, or blocks waiting for
something to happen.

Include each of these:
- any `timeout=` argument, or a `timeout` / `deadline` / `poll_interval` variable
- `asyncio.sleep`, `time.sleep`
- `await <something>.wait()`, `await queue.get()` / `dequeue()`, `asyncio.wait_for`,
  `asyncio.gather`
- `async with <lock>` / `.acquire()`
- `run_in_executor`, `subprocess.run` / `Popen`
- an HTTP client call that has no `timeout=` argument at all (record value "none")

One JSON object per occurrence:
{"line": N, "quote": "...", "kind": "timeout|sleep|event_wait|queue_wait|lock|gather|executor|subprocess|http_no_timeout",
 "value": "the literal or variable, or none",
 "enclosing_function": "name",
 "on_expiry": "what the code visibly does when this times out or fails: exception type caught and where, or 'not caught in this function'"}
