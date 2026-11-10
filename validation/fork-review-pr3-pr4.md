# Fresh-eye review of PRs 3 and 4

Reviewed PR 3: 2f8b52904819032366ccfa0a524989c7313a2268 →
557f880bc506a7cc0fe7eb3c082d4e84f00da5a6.
Reviewed PR 4: 557f880bc506a7cc0fe7eb3c082d4e84f00da5a6 →
1a116cdd1f9a317ee760ae9d080b389ca390fc43.

The fork-review skill launched a fresh read-only agent without parent conversation
history. The project-local context freeze found zero matching files. Thaw completed
and removed the lock. This is best-effort isolation; system instructions are not
erased. The reviewer did not modify the worktree.

## Fresh reviewer findings

One P2 finding in PR 4: `scripts/report_novel_information.py` unconditionally
claimed model and upstream-source hash verification, although both CLI paths
were optional. With both paths omitted, JSON reported zero checked files but
the Markdown claimed those checks had occurred. The reviewer reproduced this
with output writes intercepted in memory. The existing committed run had
actually checked six model and 21 upstream files; its measurements are unaffected.

Recommendation: merge PR 3; fix the report claim before merging PR 4.

Independent checks: 60 tests passed; all 76 PR 3 and 149 PR 4 binary hashes
matched; 21 upstream files, compatibility patch, run-script/predeclared-commit
hashes and six model files matched. The reviewer recomputed 12 PR 3 and 36 PR 4
layer restores from saved records, all matching the saved arrays. Full model
optimization and forwards were not rerun.

No issue invalidating the restoration implementation or stored results was
found. Fresh next-edit optimization, target-free greedy decoding, separate
teacher-forced diagnostics, locality failures, post-hoc word-boundary scoring,
and baseline control failures were correctly disclosed.

Scientific interpretation: exactness follows from the retained correction
information and XOR identity. Identical deterministic outputs after identical
weight restoration are an integrity check, not independent evidence of a new
learning capability. This panel does not establish general intelligence or
selective knowledge preservation; the current reports do not claim those.
Novelty cannot be established from the on-disk record alone.

## Operator annotation and resolution

Accepted the P2 finding. The report now generates its audit statement from the
actual counts and explicitly says an optional audit was not performed when its
path is omitted. Added a regression covering both omitted and supplied audits.
Regenerated the full audited report without rerunning or modifying model
measurements. All 61 tests pass. The original experiment's preregistration,
raw measurements and failed cases remain unchanged.

The limited scientific interpretation is accepted. Subsequent work must compare
existing reversal/compression/unlearning research and separate numerical rollback
from the effect of earlier edits on later optimization.
