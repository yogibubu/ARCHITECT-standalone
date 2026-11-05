# ARCHITECT standalone

ARCHITECT constructs, fits and evaluates Zoomable Asymptotically Correct Force
Fields (ZAFF). This standalone source distribution includes the coordinate,
chemical-perception, derivative and numerical dependencies required by the
ARCHITECT command. The complete nano-MATRIX checkout is not required.

## Install

Use Python 3.11 or newer and a C++17 compiler (Apple Command Line Tools on macOS,
or a C++ compiler on Linux). A Fortran compiler enables optional native kernels.
Create an isolated environment, then install all bundled components together:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python install.py
architect --help
```

Installation downloads ordinary Python dependencies and build tools from the
Python package index. The bundled MATRIX components are installed from this
repository. Optional QM programs are separately installed and licensed; their
executables and electronic calculations are not part of this distribution.

## Force-field construction and evaluation

The full command reference and executable examples are in
[the ARCHITECT manual](packages/matrix-architect/README.md). The same analytic
potential supports energy, gradient and Hessian evaluation. Coordinate and
electronic information is retained in the existing XYZIN data format.

```python
from matrix_architect import load_force_field, evaluate_force_field
field = load_force_field("molecule.zaff.json")
point = evaluate_force_field(field, coordinates_angstrom,
                             properties=("energy", "gradient", "hessian"))
```

The caller supplies `coordinates_angstrom` as an N-by-3 array. Energies are in
Hartree; Cartesian derivatives use bohr. Construction requires the input data
appropriate to the requested resolution; a transferable field and a
molecule-specific Hessian fit are distinct operations.

## Verification

```sh
python -m pip install pytest
python -m pytest -q tests/test_matrix_architect.py -k "exppe_analytic_derivatives or double_exponential_conversion or morse_analytic_derivatives or native_sparse_zaff_fit"
```

A clean installation on macOS arm64 passed `pip check`, the CLI check and five
focused scientific tests, including water force-field construction. The exact
environment and test selection are recorded in [validation.json](validation.json).
Other platforms are not covered by that validation record.
Tests that require optional electronic programs or external campaign archives
are not prerequisites for these included checks.

## Version, provenance and license

Standalone version: **0.1.0rc8**. Scientific sources are exported from
nano-MATRIX commit **d5ccb456addd65d80fbd10b693c7eaa7eca001d9**. `release.json` records the bundled component
versions and source checksums. This immutable export uses committed sources;
experimental working-copy changes are not included. The bundled GFN2 package
is a transitive LINK dependency, not a requirement to run GFN2 calculations.

The project is distributed under the BSD 3-Clause license in [LICENSE](LICENSE).
Bundled parameter data retain their original notices and licenses in the
component directories. ARCHITECT is part of nano-MATRIX, developed by Vincenzo
Barone. Please cite the accompanying ZAFF manuscript; bibliographic information
will be updated when its journal identifier is assigned.
