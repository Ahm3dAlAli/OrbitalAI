#!/bin/bash
set -e

echo "==> Stashing local changes..."
git stash

echo "==> Pulling remote changes (rebase)..."
git pull --rebase origin main

echo "==> Restoring local changes..."
git stash pop || true

echo "==> Checking for conflicts..."
if git diff --check > /dev/null 2>&1; then
    echo "No conflict markers found."
else
    echo "ERROR: Conflict markers detected. Resolve them before committing."
    git diff --check
    exit 1
fi

echo "==> Staging and committing local changes..."
if ! git diff --quiet || ! git diff --cached --quiet; then
    git add docs/
    git commit -m "update docs"
else
    echo "No local changes to commit."
fi

echo "==> Pushing to origin/main..."
git push origin main

echo "Done! Branch is published."
