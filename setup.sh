#!/usr/bin/env bash
#
# Setup script for the GJ1132 director pipeline.
# Installs all system, workspace, and Python dependencies
# needed to run: python director.py script.json
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -euo pipefail

MULTINEST_VERSION="3.10"
MULTINEST_REPO="https://github.com/JohannesBuchner/MultiNest.git"

# ── System packages ──────────────────────────────────────────────

echo "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq gfortran cmake liblapack-dev

# ── MultiNest C/Fortran library ──────────────────────────────────

if ldconfig -p | grep -q libmultinest; then
    echo "MultiNest library already installed."
else
    echo "Building MultiNest from source..."
    sBuildDir=$(mktemp -d)
    git clone --depth 1 "$MULTINEST_REPO" "$sBuildDir/MultiNest"
    mkdir -p "$sBuildDir/MultiNest/build"
    cmake -S "$sBuildDir/MultiNest" -B "$sBuildDir/MultiNest/build" \
        > /dev/null 2>&1
    make -C "$sBuildDir/MultiNest/build" -j"$(nproc)" \
        > /dev/null 2>&1 || true
    sudo cp "$sBuildDir/MultiNest/lib"/libmultinest* /usr/local/lib/
    sudo ln -sf /usr/local/lib/libmultinest.so."$MULTINEST_VERSION" \
        /usr/local/lib/libmultinest.so
    sudo ldconfig
    rm -rf "$sBuildDir"
    echo "MultiNest installed to /usr/local/lib."
fi

# ── VPLanet native binary ────────────────────────────────────────

if [ -x /workspace/vplanet-private/bin/vplanet ]; then
    echo "VPLanet binary already built."
else
    echo "Building VPLanet (optimized)..."
    make -C /workspace/vplanet-private opt
fi

# ── Workspace packages (editable installs) ───────────────────────

daEditableRepos=(
    /workspace/vplanet-private
    /workspace/vplot
    /workspace/vspace
    /workspace/multi-planet
    /workspace/bigplanet
    /workspace/vconverge
    /workspace/alabi
    /workspace/vplanet_inference
    /workspace/MaxLEV
)

echo "Installing workspace packages..."
for sRepo in "${daEditableRepos[@]}"; do
    if [ -d "$sRepo" ]; then
        sudo pip install -e "$sRepo" --quiet
    else
        echo "  Warning: $sRepo not found, skipping."
    fi
done

# ── Python dependencies ──────────────────────────────────────────

echo "Installing Python dependencies..."
pip install -r /workspace/GJ1132/requirements.txt --quiet

# ── Verify ───────────────────────────────────────────────────────

echo ""
echo "Verifying critical imports..."
python3 -c "
daModules = [
    'pymultinest', 'ultranest', 'alabi', 'vplanet',
    'vplot', 'vspace', 'multiplanet', 'bigplanet',
    'vplanet_inference', 'maxlev',
]
for sModule in daModules:
    try:
        __import__(sModule)
        print(f'  {sModule}: OK')
    except ImportError as e:
        print(f'  {sModule}: FAILED ({e})')
"

echo ""
echo "Setup complete."
