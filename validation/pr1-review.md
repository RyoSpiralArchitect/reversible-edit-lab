# PR #1 review

Reviewed head 912c9384e319c9225b57ebe93a239115b32a6580 before merge.
Checked build/undo input boundaries, logical BF16 checkpoint encoding, timing
boundaries, platform RSS units, matched checkpoint denominators, error retention,
new-directory protection, raw evidence and claims against the implementation.
No merge-blocking findings. CI run 42700188953 passed all three jobs at that head.
The 52-test suite includes source-deletion undo, corruption rejection and
identity-kernel equivalence. Process high-water RSS remains explicitly distinct
from isolated allocations; repeats remain distinct from independent seeds.

Merged using a merge commit, preserving preregistration history:
32af8bcfd77b34d52ae114936905188bd6422b4f.
