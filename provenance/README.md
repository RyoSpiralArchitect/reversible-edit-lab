# Imported evidence

Commit `b08d4e0` preserves the supplied v0.2 distribution before repository setup.
The root `MANIFEST.json` describes that imported snapshot, not the evolving HEAD.
All 51 listed file hashes were checked before making changes. The manifest itself
is the 52nd file in the import commit. The v0.1 source/results ZIP remains intact.

`results/20261110/`, `results/20261110_v02/`, and
`results/exact_chain_v02/` are historical evidence. Their source hashes refer to
the versions that produced them. New runs must use a new output directory.

`validation/github-import-unittest.txt` records a fresh run of the imported
47-test suite in an isolated virtual environment. It is separate from the
historical validation logs.

The GitHub repository adds Apache-2.0 licensing, CI, and later experiments.
Publication does not enlarge the scientific claims made by the imported work.
