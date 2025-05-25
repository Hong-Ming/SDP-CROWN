import os
import glob
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--backup', action='store_true', help='Keep a backup')
args = parser.parse_args()

START_FLAG = "### preprocessor-hint: private-section-start"
END_FLAG = "### preprocessor-hint: private-section-end"
REPLACE_FLAG = "### preprocessor-hint: private-replacement"
PRIVATE_FILE_FLAG = "### preprocessor-hint: private-file"
BACKUP_EXT = "_release_backup"


def filter(src, target):
    print(f"=== creating release file {target} from source file {src} ===")
    fs = open(src, "r")
    ft = open(target, "w")
    stop_write = False
    private_file = False
    status = "same" # same, removed, changed
    for line in fs.readlines():
        if PRIVATE_FILE_FLAG in line:
            private_file = True
            status = "removed"
            break
        if START_FLAG in line:
            stop_write = True
            status = "changed"
            continue
        if END_FLAG in line:
            stop_write = False
            continue
        if REPLACE_FLAG in line:
            line = line[line.find(REPLACE_FLAG):].replace(REPLACE_FLAG, "")
            line = line.replace("\\t", "    ")
            ft.write(line)
            continue
        if not stop_write:
            ft.write(line)
    fs.close()
    ft.close()
    if private_file:
        os.system(f"rm {target}")
    print(f"[status]: {status}")
    return


def main():
    folders = ["complete_verifier", "auto_LiRPA"]
    for folder in folders:
        if not os.path.exists(folder):
            continue
        backup_folder = f"{folder}{BACKUP_EXT}"
        if os.path.exists(backup_folder):
            print(f"{backup_folder} exists, start converting files in {backup_folder} to {folder}")
        else:
            print(f"backup {folder} to {backup_folder}")
            os.system(f"cp -r {folder} {backup_folder}")

        pyfiles = glob.glob(f"{backup_folder}/**/*.py", recursive=True)
        release_folder = folder
        for pyfile in pyfiles:
            release_file = os.path.join(release_folder, pyfile[pyfile.find("/") + 1:])
            print('!!', pyfile, release_file)
            filter(pyfile, release_file)

        if not args.backup:
            shutil.rmtree(backup_folder)

    # Delete files that we are not going to release.
    remove_list=[
        "auto_LiRPA/intermediate_refinement.py",
        "tests/gpu_tests",
        "tests/manual_tests",
        "tests/test_beta_crown.py",
        "internal_tests",
        "experimental",
        "release*",
        "Dockerfile",
        "dist",
        "build",
        "*.egg-info",
        ".pytest_cache",
        ".github",
        ".pylintrc",
        "add_copyright.py",
    ]
    for item in remove_list:
        print(f'Removing {item}')
        os.system(f"rm -rf {item}")


if __name__ == "__main__":
    main()
