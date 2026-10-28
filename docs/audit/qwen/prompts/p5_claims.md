TASK: extract every checkable claim that a docstring or comment in this file
makes about behavior, and check it only against the code in THIS file.

A checkable claim is a statement of fact about what the code does, guarantees,
never does, or how something is ordered, timed, or counted. Examples:
"never raises", "holds the lock for the duration", "processed one at a time",
"returns 409 if ...", "skipped rather than waited on", "drains the queue on failure",
any number ("timeout of 30s", "two slots").
Exclude history ("Observed on ...", "pre-fix ...") and rationale.

One JSON object per claim:
{"line": N, "quote": "the comment/docstring line containing the claim, verbatim",
 "claim": "the claim restated in one sentence",
 "about": "function or code it describes",
 "verdict": "consistent|contradicted|cannot_tell_from_this_file",
 "evidence_line": N_or_null,
 "evidence_quote": "verbatim code line supporting the verdict, or null"}

Use "contradicted" only when a specific line in this file shows otherwise, and
cite it. When the claim depends on another file, use "cannot_tell_from_this_file".
