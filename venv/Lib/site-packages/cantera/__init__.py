# This file is part of Cantera. See License.txt in the top-level directory or
# at https://cantera.org/license.txt for license and copyright information.


# start delvewheel patch
def _delvewheel_patch_1_11_2():
    import os
    if os.path.isdir(libs_dir := os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'cantera.libs'))):
        os.add_dll_directory(libs_dir)


_delvewheel_patch_1_11_2()
del _delvewheel_patch_1_11_2
# end delvewheel patch

from ._cantera import *
from ._utils import __version__, __sundials_version__, __git_commit__
from .composite import *
from .liquidvapor import *
from .onedim import *
from .utils import *
from .data import *
import cantera.interrupts  # Helps with standalone packaging (PyInstaller etc.)

from pathlib import Path
import warnings

warnings.filterwarnings('default', module='cantera')
add_data_directory(Path(__file__).parent / "data")
add_data_directory('.')  # Move current working directory to the front of the path

del warnings, Path, np, os
