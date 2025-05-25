#!/bin/bash

# DEBUG=echo
public_branch=main

echo "Release criterion:"
echo "1. All private sections deleted (will be done automatically in this script."
echo "2. All GPU tests pass and all VNN-COMP 21/22 benchmarks produce expected results."
echo "3. All command examples on README and tutorial tested working."
echo "4. Remove all code files and config files that are not part of the competition, tutorial or a paper. Remove unused and unmaintained configs! Otherwise they will confuse people since many are not updated for a long time."
echo "5. Run git status and git diff in the prerelease repo and make sure changes are updated correctly. Make sure no extra unneeded files committed."
echo
read -p "Make sure you read above carefully. Press enter to continue"

# Cleanup.
rm -rf release_abcrown/
mkdir release_abcrown/

# Clone both repositories.
internal=release_abcrown/internal
public=release_abcrown/public
new=release_abcrown/new
current=$(pwd)
git clone git@github.com:Verified-Intelligence/Verifier_Development.git $internal
git clone git@github.com:Verified-Intelligence/alpha-beta-CROWN.git $public
# Switch branch
pushd $public
# The automatic script only releases to an internal prerelease repository
git remote set-url origin git@github.com:Verified-Intelligence/alpha-beta-CROWN-prerelease.git
git checkout ${public_branch}
popd

# Make a directory.
mkdir $new

# Copy other files from the internal repository.
cp -va "$internal/." $new
# Override with .git from the public repository.
rm -rf $new/.git
cp -r $public/.git $new

# Prepare for release
pushd $new
python $current/release_preprocessor.py
rm .readthedocs.yaml setup.py
rm -rf internal_tests doc tests examples
mv README_abcrown.md README.md
cp $current/complete_verifier/LICENSE .
cp -r $current/$public/.github .
cp -r $current/$public/.gitmodules .

rm -rf complete_verifier/exp_configs/full_dataset
rm -rf complete_verifier/exp_configs/bab_attack/attack_ubs
rm -rf complete_verifier/exp_configs/bab_attack/_unused
rm -rf complete_verifier/exp_configs/vnncomp22/before_submission
rm -rf complete_verifier/models/nonlinear_NNs
rm -rf complete_verifier/benchmark_data
rm -rf complete_verifier/cuts/CPLEX_cuts/README.md

git rm -rf auto_LiRPA
rm -rf auto_LiRPA
git submodule add https://github.com/Verified-Intelligence/auto_LiRPA.git
pushd complete_verifier
python arguments.py # To update abcrown_all_params.yaml
rm auto_LiRPA
ln -s ../auto_LiRPA/auto_LiRPA .
popd

echo "Changed files:"
git add .
git commit -m 'Jan 2025 prerelease'
git push -f
popd
