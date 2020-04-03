import os
import sys
import cv2
import csv
import numpy as np
import openslide
import math
import argparse

import matplotlib as mpl
mpl.use('TkAgg')
mpl.use('pdf')
import matplotlib.pylab as plt

from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
import multiprocessing


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')