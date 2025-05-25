#!/bin/bash

# DEBUG=echo
private_branch=master
public_branch=master

# Cleanup.
rm -rf release/
mkdir release/

# Clone both repositories.
internal=release/internal
public=release/public
new=release/new
current=$(pwd)
git clone git@github.com:Verified-Intelligence/Verifier_Development.git $internal
git clone git@github.com:Verified-Intelligence/auto_LiRPA.git $public
# Switch branch
pushd $public
# The automatic script only releases to an internal prerelease repository
git remote set-url origin git@github.com:Verified-Intelligence/auto_LiRPA-prerelease.git
git checkout ${public_branch}
popd

# Make a directory.
mkdir $new

# Copy other files from the internal repository.
cp -a "$internal/." $new
# Override with .git from the public repository.
rm -rf $new/.git
cp -r $public/.git $new

# Prepare for release
pushd $new
rm -rf complete_verifier
rm -rf vnncomp_scripts
rm -rf internal_tests
python $current/release_preprocessor.py
rm README_abcrown.md
cp -r $current/$public/.github .
echo "Changed files:"
git add .
git commit -m 'Jan 2025 prerelease'
git push -f
popd
