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
#   4. check that hyp1.40, hypoDD and ph2dt are available
#
# It never edits your shell configuration files (~/.bashrc, ~/.zshrc).
# Set QQ_BIN_DIR to check another folder than ~/bin in step 4.
# =================================================================
set -euo pipefail

ENV_NAME="${1:-quickquake}"
BIN_DIR="${QQ_BIN_DIR:-$HOME/bin}"
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

echo "==> 4/4 HypoInverse and HypoDD programs in $BIN_DIR"
missing=0
for program in hyp1.40 hypoDD ph2dt; do
    if [ -x "$BIN_DIR/$program" ]; then
        echo "    found    $BIN_DIR/$program"
    else
        echo "    MISSING  $BIN_DIR/$program"
        missing=1
    fi
done

echo ""
echo "================================================================"
if [ "$missing" -eq 0 ]; then
    echo "Installation complete. Try the example project:"
else
    echo "Python part installed. Before running the location step, compile the"
    echo "missing programs (README: 'Installing HypoInverse and HypoDD') and put"
    echo "them in $BIN_DIR, or set location_binpath in your QQ_driver.py to their folder."
    echo ""
    echo "Then try the example project:"
fi
echo "  conda activate $ENV_NAME"
echo "  python examples/pavlof_2021/QQ_driver.py"
echo "================================================================"
