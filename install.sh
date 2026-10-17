#!/usr/bin/env bash
# =================================================================
# QuickQuake installation (macOS and Linux)
#
#   bash install.sh [environment-name]        (default name: quickquake)
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

echo "==> 1/4 Conda environment '$ENV_NAME'"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "    already exists, not recreated (remove it with: conda env remove -n $ENV_NAME)"
else
    "$CONDA" env create -n "$ENV_NAME" -f environment.yml
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
