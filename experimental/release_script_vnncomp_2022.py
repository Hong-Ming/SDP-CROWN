import os

BACKUP_DIR = "../alpha-beta-CROWN_vnncomp22"


def main():

    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

    os.system(f"cp -rf auto_LiRPA {BACKUP_DIR}")

    if not os.path.exists(os.path.join(BACKUP_DIR, 'complete_verifier/exp_configs/vnncomp22')):
        os.makedirs(os.path.join(BACKUP_DIR, 'complete_verifier/exp_configs/vnncomp22'))

    os.system(f"cp -rf complete_verifier/exp_configs/vnncomp22  {os.path.join(BACKUP_DIR, 'complete_verifier/exp_configs')}")
    os.system(f"cp -rf vnncomp_scripts  {os.path.join(BACKUP_DIR)}")

    selected_files = ['arguments.py', 'attack', 'abcrown.py',
                      'batch_branch_and_bound.py', 'batch_branch_and_bound_input_split.py', 'beta_CROWN_solver.py',
                      'branching_domains.py', 'branching_domains_input_split.py', 'environment_vnncomp22.yml',
                      'branching_heuristics.py', 'convert_nn4sys_model.py', 'cuts/cut_utils.py', 'cuts/implied_cuts.py',
                      'lp_mip_solver.py', 'model_defs.py', 'nn4sys_verification.py', 'onnx_opt.py', 'optimized_cuts.py', 'read_vnnlib.py',
                      'requirements.txt', 'scip_model.py', 'split_onnx.py', 'tensor_storage.py', 'utils.py',
                      'vnncomp_main_2022.py', 'jit_precompile.py',
                      'cuts/CPLEX_cuts']

    for file in selected_files:
        os.system(
            f"cp -rf complete_verifier/{file}  {os.path.join(BACKUP_DIR, 'complete_verifier')}")

    # remove __pycache__
    os.system(f'find {BACKUP_DIR} | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf')
    # remove before_submission folder
    os.system(f"rm -r {os.path.join(BACKUP_DIR, 'complete_verifier/exp_configs/vnncomp22/before_submission')}")


if __name__ == "__main__":
    main()
