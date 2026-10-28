You are doing a mechanical inventory of source code for a later human-led audit.
You are NOT judging whether code is correct, and you are NOT proposing fixes.
Your output is a checklist that will be verified line by line against the source.

Rules:
- The source is given with line numbers in the form `NNNN| code`. Cite the line
  number exactly as printed. Never estimate or count lines yourself.
- For every entry, `quote` must be copied VERBATIM from that one line (without
  the `NNNN| ` prefix). Entries whose quote does not appear on the cited line
  are discarded, so do not paraphrase.
- Be exhaustive. A missed item is worse than an extra one.
- If a field cannot be determined from the code shown, write "unknown". Do not
  guess about code you have not been shown.
- Output ONLY JSON Lines: one JSON object per line, no prose, no markdown
  fences, no trailing summary. If there are no entries, output exactly: {"none": true}
