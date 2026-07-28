#!/bin/bash
# Group the working-tree changes into logical commits (one clear message each) and
# push. Safe by construction: it NEVER stages model weights (*.pt), datasets (*.npy),
# or the data/output folders — only source, scripts, docs, and results.
#
#   bash scripts/push_grouped.sh            # commit each group, then push
#   DRY=1 bash scripts/push_grouped.sh      # show what WOULD be committed, no changes
#
# Edit the parallel MSGS[] / SPECS[] arrays below to taste (same index = one commit).
set -u
cd "$(dirname "$0")/.."
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "[err] not a git repo"; exit 1; }

DRY="${DRY:-}"
TRAILER="Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
# Hard exclusions applied to EVERY group (belt-and-suspenders over .gitignore).
EXCL=(':(exclude)*.pt' ':(exclude)*.npy' ':(exclude)*.pth'
      ':(exclude)OrbitSight_Dataset/**' ':(exclude)OrbitSight_dataset/**'
      ':(exclude)OrbitalAI/**' ':(exclude)docker_out/**' ':(exclude)mini/**'
      ':(exclude)predictions/**' ':(exclude)*.xlsx')

# --- logical groups: MSGS[i] is committed with the paths in SPECS[i] -------- #
MSGS=(
  "core: model architecture + loss changes"
  "scripts: training / inference / tooling"
  "docker: deployment image + entrypoint"
  "docs: paper / proposal / README / roadmap"
  "results + figures"
  "config / environment"
)
SPECS=(
  "orbitsight/"
  "scripts/"
  "Dockerfile run_infer.sh run.sh .dockerignore requirements.txt"
  "docs/*.md docs/*.html docs/*.pdf README.md ROADMAP.md Data.md"
  "results.json docs/figures/"
  "environment.yml"
)

commit_group() {
  local msg="$1"; shift
  local specs=("$@")   # specific paths only; excludes break `git add` on untracked
  if [ -z "$(git status --porcelain -- "${specs[@]}" 2>/dev/null)" ]; then
    echo "  [skip] no changes    — $msg"; return
  fi
  if [ -n "$DRY" ]; then
    echo "  [DRY ] would commit  — $msg"
    git status --porcelain -- "${specs[@]}" 2>/dev/null | sed 's/^/           /'
    return
  fi
  git add -- "${specs[@]}"
  # safety net: never let a weight/data/output file slip in, even if a spec is broadened
  git reset -q -- '*.pt' '*.pth' '*.npy' '*.xlsx' OrbitSight_Dataset OrbitalAI docker_out mini predictions 2>/dev/null || true
  if git diff --cached --quiet; then echo "  [skip] nothing staged — $msg"; return; fi
  git commit -q -m "$msg" -m "$TRAILER"
  echo "  [commit] $msg"
}

echo "== grouping changes into commits =="
for i in "${!MSGS[@]}"; do
  # shellcheck disable=SC2086
  commit_group "${MSGS[$i]}" ${SPECS[$i]}
done

# Report anything left uncommitted (excluding the hard-excluded data/models).
LEFT="$(git status --porcelain -- . "${EXCL[@]}" 2>/dev/null)"
if [ -n "$LEFT" ]; then
  echo ""
  echo "== NOT committed (no group matched — add a group or handle manually) =="
  echo "$LEFT" | sed 's/^/   /'
fi

echo ""
[ -n "$DRY" ] && { echo "== DRY run: no commits made, nothing pushed =="; exit 0; }

UNPUSHED="$(git log --oneline @{u}..HEAD 2>/dev/null || git log --oneline origin/main..HEAD 2>/dev/null)"
[ -z "$UNPUSHED" ] && { echo "== nothing to push (up to date) =="; exit 0; }
echo "== pushing:"; echo "$UNPUSHED" | sed 's/^/   /'
git push && echo "== done =="
