#!/usr/bin/env bash
# Copy `main` and the tags from github.com to the read-only mirror on the
# A*STAR GitHub Enterprise Server (GHES).
#
# Why a script and not a GitHub Actions workflow: the GHES refuses connections
# from GitHub's hosted runners (HTTP 403 before any login), so the copy must be
# pushed from a machine that the GHES accepts.
#
# Requirements: git, and the GitHub command line tool `gh` logged in to the
# GHES (`gh auth login --hostname ghes.a-star.edu.sg`).
#
# The script keeps its own bare repository, so it does not depend on the
# branch or the state of any working checkout. The push is never forced: if
# someone commits on the GHES copy, the script fails and leaves it untouched.
#
# Environment:
#   MLIPRUN_MIRROR_DIR     where the bare repository lives
#                          (default: ~/.cache/mliprun-ghes-mirror.git)
#   MLIPRUN_MIRROR_NOTIFY  set to 1 to show a macOS notification on failure
#                          (used by the scheduled job; a silent failure there
#                          would leave the copy stale without anyone knowing)
set -euo pipefail

GITHUB_URL="https://github.com/manuelarcer/mliprun.git"
GHES_URL="https://ghes.a-star.edu.sg/juar/mliprun.git"
MIRROR_DIR="${MLIPRUN_MIRROR_DIR:-$HOME/.cache/mliprun-ghes-mirror.git}"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

on_failure() {
  echo "$(stamp) FAILED: the GHES copy was not updated (exit code $1)"
  if [ "${MLIPRUN_MIRROR_NOTIFY:-0}" = "1" ] && command -v osascript >/dev/null 2>&1; then
    osascript -e 'display notification "The GHES copy was not updated. See ~/Library/Logs/mliprun-ghes-mirror.log" with title "mliprun: GHES mirror failed"' || true
  fi
}
trap 'on_failure $?' ERR

for tool in git gh; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$(stamp) '$tool' was not found on PATH"
    false
  fi
done

# Use the gh login for the GHES, and nothing else, as the git credential.
ghes_git() {
  git -C "$MIRROR_DIR" -c credential.helper= \
    -c 'credential.helper=!gh auth git-credential' "$@"
}

if [ ! -d "$MIRROR_DIR" ]; then
  git init --bare --quiet "$MIRROR_DIR"
fi

# github.com is public, so the fetch needs no login.
git -C "$MIRROR_DIR" fetch --quiet --tags "$GITHUB_URL" "+refs/heads/main:refs/heads/main"
ghes_git push --quiet "$GHES_URL" refs/heads/main --tags

# Confirm the result instead of trusting the exit code of the push.
want=$(git -C "$MIRROR_DIR" rev-parse refs/heads/main)
have=$(ghes_git ls-remote "$GHES_URL" refs/heads/main | cut -f1)
if [ "$want" != "$have" ]; then
  echo "$(stamp) GHES main is ${have:-missing}, expected $want"
  false
fi
echo "$(stamp) OK: GHES main is ${want:0:7}, same as github.com"
