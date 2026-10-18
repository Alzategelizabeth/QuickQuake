#!/usr/bin/env bash
# =================================================================
# QuickQuake installation (macOS and Linux)
#
#   bash install.sh [environment-name]            (default name: quickquake)
#   bash install.sh environment-name --update     (reinstall into an existing environment)
#
# An environment that already exists is never removed or changed unless you add --update.
#
# Steps:
#   1. create the conda environment from environment.yml
#   2. download HypoXPy (git submodule) if the folder is empty
#   3. install PhaseNet, GaMMA and HypoXPy from dependencies/
#   4. compile HypoInverse (hyp1.40), HypoDD and ph2dt with the compilers of the
#      environment and put the programs in the environment (no ~/bin needed)
#
# It never edits your shell configuration files (~/.bashrc, ~/.zshrc) and
# writes only inside this folder and the conda environment.
# =================================================================
set -euo pipefail

ENV_NAME="${1:-quickquake}"
UPDATE="${2:-}"                  # "--update" = install into an environment that already exists
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if command -v mamba >/dev/null 2>&1; then
    CONDA=mamba
elif command -v conda >/dev/null 2>&1; then
    CONDA=conda
else
    echo "conda was not found. Install Miniforge first: https://github.com/conda-forge/miniforge"
    exit 1
fi

# Does an environment with this name already exist? Upper/lower case are ignored, because
# macOS folder names ignore them: "quickquake" and "QuickQuake" are the same folder.
env_exists() {
    if [ "$(echo "$ENV_NAME" | tr '[:upper:]' '[:lower:]')" = "base" ]; then return 0; fi
    # registered environments (their names and folder names), in any conda installation
    if conda env list 2>/dev/null | grep -v '^#' | awk 'NF {print $1; print $NF}' |
       while read -r path; do basename "$path"; done | grep -qix -- "$ENV_NAME"; then
        return 0
    fi
    # a folder of that name in conda's environment folders
    for dir in $(conda config --show envs_dirs 2>/dev/null | awk '/^  - / {print $2}'); do
        if [ -e "$dir/$ENV_NAME" ]; then return 0; fi
    done
    return 1
}

echo "==> 1/4 Conda environment '$ENV_NAME'"
if env_exists; then
    if [ "$UPDATE" != "--update" ]; then
        echo "    An environment called '$ENV_NAME' (in some upper/lower case) already exists."
        echo "    Nothing was changed. To install QuickQuake next to it, choose another name:"
        echo "      bash install.sh quickquake_new"
        echo "    To reinstall into that same environment (PhaseNet, GaMMA, HypoXPy and the"
        echo "    compiled programs are replaced; its other packages are not touched):"
        echo "      bash install.sh $ENV_NAME --update"
        exit 1
    fi
    echo "    already exists: updating it (--update)"
else
    # -y answers "Confirm changes [Y/n]". With -y conda also deletes an existing environment
    # of the same name, which is why env_exists() is checked first.
    "$CONDA" env create -y -n "$ENV_NAME" -f environment.yml
fi

echo "==> 2/4 HypoXPy source"
if [ ! -f dependencies/HypoXPy/setup.py ]; then
    git submodule update --init dependencies/HypoXPy
fi

echo "==> 3/4 Installing PhaseNet, GaMMA and HypoXPy into '$ENV_NAME'"
conda run -n "$ENV_NAME" python -m pip install -e dependencies/PhaseNet -e dependencies/GaMMA
conda run -n "$ENV_NAME" python -m pip install dependencies/HypoXPy
# pip leaves build folders inside HypoXPy; the installed copy does not need them
rm -rf dependencies/HypoXPy/build dependencies/HypoXPy/*.egg-info

echo "==> 4/4 Compiling HypoInverse (hyp1.40), HypoDD and ph2dt into '$ENV_NAME'"
ENV_BIN="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')/bin"

# Compilers installed by environment.yml (gfortran, c-compiler, make). On Linux they
# carry a prefix, e.g. x86_64-conda-linux-gnu-gfortran.
first_program() {            # print the first of the given names found in the environment
    for name in "$@"; do
        for path in "$ENV_BIN"/$name; do
            if [ -x "$path" ]; then echo "$path"; return; fi
        done
    done
}
FC_BIN="$(first_program gfortran '*-gfortran')"
CC_BIN="$(first_program clang gcc '*-clang' '*-gcc' cc)"
MAKE_BIN="$(first_program make)"
if [ -z "$FC_BIN" ] || [ -z "$CC_BIN" ] || [ -z "$MAKE_BIN" ]; then
    echo "    The environment has no Fortran compiler, C compiler or make. Add them with:"
    echo "      conda install -n $ENV_NAME -c conda-forge gfortran c-compiler make"
    echo "    and run this script again."
    exit 1
fi

# Build in a temporary folder from the HypoXPy sources stored in git (not from the
# working folder): old object files (*.o) are never reused and dependencies/ stays clean.
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT
git -C dependencies/HypoXPy archive HEAD external | tar -x -C "$BUILD_DIR"
SRC="$BUILD_DIR/external"
# The HypoInverse makefile copies hyp1.40 to ~/bin: HOME points to the build folder
# during that step, so your real ~/bin is never touched.
mkdir -p "$BUILD_DIR/home/bin"
export PATH="$(dirname "$FC_BIN"):$PATH"
if command -v xcrun >/dev/null 2>&1; then      # macOS: where the system libraries are
    export SDKROOT="$(xcrun --show-sdk-path)"
fi
LOG="$BUILD_DIR/compile.log"
if ! { HOME="$BUILD_DIR/home" "$MAKE_BIN" -C "$SRC/hypoinverse1.40/source" FC="$FC_BIN" hyp1.40 &&
       "$MAKE_BIN" -C "$SRC/hypodd2.1/src/hypoDD" FC="$FC_BIN" CC="$CC_BIN" &&
       "$MAKE_BIN" -C "$SRC/hypodd2.1/src/ph2dt" FC="$FC_BIN" CC="$CC_BIN"; } > "$LOG" 2>&1; then
    cp "$LOG" "$REPO_DIR/compile_error.log"
    tail -20 "$LOG"
    echo "    Compilation failed. The full log is in compile_error.log"
    exit 1
fi
cp "$SRC/hypoinverse1.40/source/hyp1.40" "$SRC/hypodd2.1/src/hypoDD/hypoDD" \
   "$SRC/hypodd2.1/src/ph2dt/ph2dt" "$ENV_BIN/"
for program in hyp1.40 hypoDD ph2dt; do
    echo "    installed $ENV_BIN/$program"
done

echo ""
echo "================================================================"
echo "Installation complete. Try the example project:"
echo "  conda activate $ENV_NAME"
echo "  python examples/pavlof_2021/QQ_driver.py"
echo "================================================================"
