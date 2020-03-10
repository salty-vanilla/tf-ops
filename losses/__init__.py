import os
import sys
import pathlib

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(os.path.join(current_dir.parent))

from losses.gan import *
from losses.mmd import *
from losses.image import *