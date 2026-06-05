#!/usr/bin/env bash
# clean_install.sh — clean reinstall for blg_model_builder_v2 (pure Python)
#
# No C++ compilation required.  All interatomic potential calculators use the
# LAMMPS Python module.  To install the LAMMPS Python module, build LAMMPS
# with Python support and run `make install-python`:
#
#   cd $LAMMPS_ROOT/build
#   cmake ../cmake -DLAMMPS_EXCEPTIONS=yes -DBUILD_SHARED_LIBS=yes \
#       -DPKG_INTERLAYER=yes -DPKG_MANYBODY=yes -DPKG_ML-POD=yes \
#       -DPKG_MOLECULE=yes -DWITH_PYTHON=yes
#   cmake --build . -- -j$(nproc)
#   make install-python
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

PKG=blg-model-builder-v2

echo "=== Uninstalling $PKG ==="
pip uninstall "$PKG" -y 2>/dev/null || true

echo "=== Removing stale build artifacts ==="
find . -name "*.egg-info" -type d -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
find . -name "__pycache__"  -type d -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
rm -rf build/

echo "=== Installing ==="
pip install -e .

echo "Successfully installed $PKG"
echo ""
echo "NOTE: Ensure the LAMMPS Python module is installed (make install-python)."
