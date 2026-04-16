# Releasing Quartermaster

## ⚠️ The trap (don't do this)

The `Publish to PyPI` workflow runs on whatever branch you click "Run workflow" from. It does **NOT** merge `develop` into `master` first. If you click Publish from `master` while the actual changes still live on `develop`, PyPI gets a release containing only the version-bump diff. **This happened with v0.1.2.** The release was empty; downstream integrators were blocked.

## The release flow

Every release crosses two branches: `develop` (where work lands) and `master` (where releases ship from). PyPI publishes from master. Therefore:

```
feature branch  →  PR → develop  →  release PR → master  →  click Publish
```

### Step 1 — work merges to `develop`

Open feature/fix PRs against `develop`. CI runs there. Merge when green. **Master is untouched at this point.**

### Step 2 — open a release PR `develop → master`

```bash
gh pr create --base master --head develop \
  --title "Release vX.Y.Z: <one-line summary>" \
  --body "Brings master in line with develop. After merge, click Publish to PyPI with bump=patch (or minor/major)."
```

Get this PR reviewed and merged. **Master now contains the actual code that will ship.** The publish workflow itself does *not* perform this merge.

### Step 3 — click "Run workflow" on `Publish to PyPI`

[Actions → Publish to PyPI → Run workflow], pick `bump=patch` (or whatever level fits the changes), branch=`master`, `dry_run=false`. The workflow:

1. Reads the last git tag on master (e.g. `v0.1.2`).
2. Bumps it (`patch` → `0.1.3`).
3. Updates `version = "X.Y.Z"` in every `pyproject.toml` and the SDK's `__init__.py`.
4. Updates cross-package dependency pins (`quartermaster-foo>=X.Y.Z`).
5. Commits as `Release vX.Y.Z`, tags `vX.Y.Z`, pushes.
6. Builds wheels in matrix per package.
7. Publishes to PyPI in dependency order (providers → graph → tools → nodes → engine → mcp-client → code-runner → sdk).

### Step 4 — sync `develop` back

After publish, master will have the auto-generated `Release vX.Y.Z` commit that develop doesn't. Bring develop back in line:

```bash
git checkout develop
git pull origin develop
git merge origin/master --no-edit   # picks up the release-bump commits
git push origin develop
```

This step is what makes the *next* release PR (`develop → master`) clean — without it, develop slowly drifts behind on version bumps and you get noisy merge conflicts.

### Step 5 — open a release-notes Discussion

Every release gets its own thread in [Discussions → General](https://github.com/MindMadeLab/quartermaster-sdk-py/discussions/categories/general). This is where users find the changelog, known issues, and migration notes — the PR description and tag annotations are not enough.

Create one thread per release with:

- **Title:** `Release vX.Y.Z — <one-line headline>` (e.g. `Release v0.3.1 — End-node revert, .back() node, telemetry auto-import`)
- **Body sections:** TL;DR with `pip install` line · what shipped (table) · breaking changes / known issues · migration notes (from previous version + from earlier majors) · test count delta · "what we learned" if there's a story to tell · open questions for the community

```bash
# Drop the body in /tmp/release_vX.Y.Z.md, then:
gh api graphql \
  -F repoId="R_kgDOPGPFkQ" \
  -F catId="DIC_kwDOPGPFkc4C64Zv"  `# General category` \
  -F title="Release vX.Y.Z — <headline>" \
  -F body="@/tmp/release_vX.Y.Z.md" \
  -f query='mutation($repoId:ID!,$catId:ID!,$title:String!,$body:String!){
    createDiscussion(input:{
      repositoryId:$repoId,
      categoryId:$catId,
      title:$title,
      body:$body
    }){discussion{number url}}
  }'
```

Cross-link adjacent releases (the `vX.Y.Z` thread should mention `vX.Y.(Z-1)` for migration context, and vice versa for "did you mean to install the newer one?"). Edit existing threads with `updateDiscussion(input:{discussionId:..., body:...})`.

If the release ships a known wart that warrants users skipping the version, say so in the TL;DR and pin the next release's thread above it. (Examples: [#21 v0.3.0](https://github.com/MindMadeLab/quartermaster-sdk-py/discussions/21) → [#22 v0.3.1](https://github.com/MindMadeLab/quartermaster-sdk-py/discussions/22).)

## Common mistakes

| Symptom | Cause | Fix |
|---|---|---|
| **PyPI shipped empty packages** (just version bumps, no code) | Clicked Publish on master without first merging `develop → master`. | Open a real release PR (Step 2), merge, click Publish again. The yanked-empty release stays as `vX.Y.Z`; the real one becomes `vX.Y.Z+1`. |
| **`Release vX.Y.Z` commit conflicts when merging develop into master** | Develop didn't get the previous release's bump commit synced back (Step 4 was skipped). | Resolve conflicts by taking master's bumped version strings; commit the merge. |
| **Publish failed mid-way, only some packages on PyPI** | One sub-package's wheel build or `twine upload` failed; later packages in the dependency chain shipped without their dependency. | Manually publish the missing sub-packages with `twine upload dist-PACKAGE/*` from a local checkout of the `vX.Y.Z` tag. |
| **`pip install quartermaster-sdk==X.Y.Z` fails** with `No matching distribution found for quartermaster-nodes>=X.Y.Z` | Sub-package upload didn't complete before SDK upload. | Same as above — finish the sub-package upload, then `pip install` will succeed. |

## Hotfix flow (skip develop)

For urgent master-only fixes:

```bash
git checkout master
git pull
git checkout -b hotfix/vX.Y.Z+1-<topic>
# … fix, commit …
gh pr create --base master --title "Hotfix: ..." --body "..."
# merge, then Step 3 (Publish), then Step 4 (sync develop)
```

**Hotfix PRs still need at least one review approval before merge** — the Publish workflow has no minimum-approval gate of its own (`RELEASE_PAT` bypasses the admin ruleset), so the PR review is the only thing standing between an unreviewed commit and a public PyPI release. Don't self-approve hotfixes; ping a co-maintainer.

Same Step 4 caveat — sync develop back from master so the hotfix isn't lost on the next regular release.
