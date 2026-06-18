# GitHub Actions workflows

Two workflows drive CI and releases for PyChunkedGraph.

| File | Workflow name | Trigger | Purpose |
|---|---|---|---|
| `main.yml` | PyChunkedGraph | push / PR to `main` or `pcgv3` | Build the image and run the test suite with coverage |
| `release.yml` | publish release | manual (`workflow_dispatch`) | Bump the version, tag, and create the GitHub Release; optionally bump the Helm chart |

## `main.yml` — CI

Runs on every push and pull request targeting `main` or `pcgv3`. One `unit-tests` job:

1. Builds the Docker image locally with Buildx (`load: true`, not pushed), tagged with the commit SHA, using the GitHub Actions layer cache.
2. Runs `pytest` with coverage inside the container against `pychunkedgraph/tests`.
3. Copies `coverage.xml` out and uploads it to Codecov — runs even when tests fail, and a Codecov upload error does not fail the build.
4. Removes the test container.

Secrets: `CODECOV_TOKEN`.

## `release.yml` — release

Manual only — dispatch it from the branch you want to release. **The branch is the major line**: `main` carries 2.x, `pcgv3` carries 3.x, because the version is a committed literal in `pychunkedgraph/_version.py` (the repo-root README has the full versioning flow). The tag and the in-code literal are written by the same job, so they never drift.

### Inputs

| Input | Default | Effect |
|---|---|---|
| `part` | `patch` | which semver component to bump — `major` / `minor` / `patch` |
| `dry-run` | `false` | `true` computes the next version and stops: no commit, tag, release, or chart bump |
| `skip-tests` | `false` | currently unused — no step references it |
| `update-chart` | `false` | `true` also bumps the Helm chart `appVersion` (needs `HELM_CHART_UPDATE_TOKEN`) |

### Jobs

- **`bump`** — reads the version from `pychunkedgraph/_version.py`, bumps `part`, writes it back. Unless dry-run: commits `release vX.Y.Z`, tags `vX.Y.Z`, pushes the branch and the tag, then creates the GitHub Release. Needs `contents: write`.
- **`update-chart`** — opt-in: runs only when `update-chart` is `true` (and not dry-run). Checks out `CAVEconnectome/cave-helm-charts`, sets `charts/pychunkedgraph/Chart.yaml` `appVersion` to the new version, bumps the chart's own `version` by a patch, and pushes. Needs `HELM_CHART_UPDATE_TOKEN` (write access to the chart repo).

## Cutting a release

```
# preview the next version, no writes
gh workflow run release.yml --ref pcgv3 -f part=patch -f dry-run=true

# cut the release
gh workflow run release.yml --ref pcgv3 -f part=patch
```

Or via the Actions UI: **publish release → Run workflow →** pick the branch and `part`.

The pushed tag is what the image build (`cloudbuild.yaml`) builds from. New tables created by that image are stamped with this version, and the server only serves tables whose major matches. The Helm chart `appVersion` bump that rolls the image out is opt-in via `update-chart`.
