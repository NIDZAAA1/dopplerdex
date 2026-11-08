# Contributing to dopplerdex

Thanks for helping! Bug reports, test recordings with known locations (described, not uploaded),
detector and search improvements, documentation and UI work are all welcome.

## The one hard rule: no recordings, no private locations

**Never commit audio or video files, or the location where a private recording was made.**
This includes your own recordings. Recordings can identify people and places; this repository
contains code and openly licensed metadata only.

- `.gitignore` excludes every common audio/video format, `data/`, `.env`, and any `TRUTH*`,
  `*.local.json` or `*.private.*` file.
- `scripts/check_repo_hygiene.py` checks file *contents* (a WAV renamed to `.jpeg` is still caught),
  file sizes, private-truth file names and personal absolute paths. It runs in CI on every pull request
  and as a pre-commit hook (see below). A pull request that fails it cannot be merged.
- Benchmarks reference public recordings by URL and licence in a `manifest.json`; the audio is
  downloaded locally by the benchmark scripts. If a ground-truth answer is not public (e.g. a challenge),
  keep it in a gitignored `*.local.json` file.

## Development setup

```bash
uv sync                                   # Python 3.13 + dependencies + dev tools
uv run pre-commit install                 # hygiene check, ruff and formatting before each commit
uv run pytest                             # tests
uv run ruff check . && uv run ruff format --check .
```

Most tests are fast and offline. A few download small reference files (aircraft type table, geoid)
on first run and skip when offline.

## How to contribute

1. Open an issue first for anything bigger than a small fix, so we can agree on the approach.
2. Fork, create a branch from `main` (`feat/...`, `fix/...`, `docs/...`).
3. Keep changes focused; add or update tests. Physics and scoring changes need a test that would have
   failed before the change.
4. **Improvements must be general.** Do not tune thresholds to one recording. If a change helps one
   benchmark case, show it does not hurt the others (`bench/`), and describe the results in the PR.
5. Run the checks above, then open a pull request against `main`. CI must pass and one maintainer
   review is required. `main` is protected: no direct pushes, no force pushes.

## Commit messages

Short imperative subject line (`Fix wind sign in retarded-time solver`), blank line, then what and
why. Reference issues with `#123`.

## Reporting bugs

Use the bug report template. Include the command or web-app steps, the full error, OS and Python
version, and (if relevant) the detector output (`analyze_recording.py` report). Do **not** attach
private recordings; describe them (length, device, environment) or share a public, openly licensed one.

## Code style

- Python 3.13, type hints on public functions, docstrings that explain the physics or the reasoning.
- `ruff` for linting and formatting (config in `pyproject.toml`).
- Prefer clear, vectorised NumPy; use numba only for proven hot loops.

## Licence of contributions

dopplerdex is GPL-3.0-or-later. By submitting a contribution you agree that it is licensed under the
same terms, and that you have the right to submit it.

## Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind and constructive.
