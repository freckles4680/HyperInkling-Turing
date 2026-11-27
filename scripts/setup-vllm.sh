#!/bin/bash
# Recreate ./vllm: vllm-project/vllm at the commit in patches/BASE (the head of PR
# #69165, Inkling 2 on SM80/86/89, one commit on main), plus HyperInkling's Turing
# (sm75) port (patches/vllm-hyperinkling-sm75.patch) as one commit on branch
# hyperinkling/sm75. Then build with scripts/build-vllm.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
base=$(cat patches/BASE)
if [ ! -d vllm/.git ]; then
	git clone --filter=blob:none https://github.com/vllm-project/vllm.git vllm
fi
cd vllm
# By SHA first (GitHub serves any commit it still has); the PR ref only if that
# fails. A force-pushed PR no longer has BASE: then move BASE and rebase the patch.
git fetch --filter=blob:none origin "$base" || git fetch origin pull/69165/head
git cat-file -e "$base^{commit}" 2>/dev/null \
	|| { echo "BASE $base is gone from PR #69165: rebase the patch" >&2; exit 1; }
# An existing checkout must be clean, or the checkout below refuses. A drive that cannot
# store the executable bit (exFAT, NTFS, some network mounts) makes every file look
# modified ("mode change 100644 => 100755"), and a stash does not help there: git cannot
# chmod the files back. Mode-only differences are then ignored for this repository.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
	if [ -z "$(git -c core.fileMode=false status --porcelain --untracked-files=no)" ]; then
		echo "vllm/: only file modes differ (this drive does not keep them): core.fileMode false"
		git config core.fileMode false
	fi
	if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
		echo "vllm/ has local changes; keep them with  git -C vllm stash  and rerun:" >&2
		git status --short --untracked-files=no | head -20 >&2
		exit 1
	fi
fi
git checkout -B hyperinkling/sm75 "$base"
git am --3way --keep-non-patch ../patches/vllm-hyperinkling-sm75.patch
git log --oneline -2
