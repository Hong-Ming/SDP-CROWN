#!/bin/bash

# DEBUG=echo
private_branch=master
public_branch=master

# Cleanup.
rm -rf release/
mkdir release/

internal=release/internal
new=release/new
current=$(pwd)
git clone git@github.com:Verified-Intelligence/Verifier_Development.git $internal

# Make a directory.
mkdir $new

# Copy other files from the internal repository.
cp -va "$internal/." $new

# Prepare for release
pushd $new
rm -rf .git
rm -rf internal_tests
rm -rf experimental
rm -rf .github
rm -rf tests
rm add_copyright.py
rm Dockerfile
rm release_*

git init
git remote add origin git@github.com:Verified-Intelligence/alpha-beta-CROWN_vnncomp2024.git
git add .
git commit -m 'VNNCOMP submission'
git push origin master -f
popd
