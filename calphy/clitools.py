"""
calphy: a Python library and command line interface for automated free
energy calculations.

Copyright 2021-2026 (c) Sarath Menon, Yury Lysogorskiy, Ralf Drautz
Interdisciplinary Centre for Advanced Materials Simulation (ICAMS),
Ruhr University Bochum, 44801 Bochum, Germany

calphy is published and distributed under the Academic Software Licence v1.0 (ASL).
calphy is distributed in the hope that it will be useful for non-commercial academic
research, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the LICENSE file for details.

The ASL permits academic non-commercial use only. Contact
sarath.menon@ruhr-uni-bochum.de to enquire about commercial use rights.

More information about the program can be found in:
Menon, Sarath, Yury Lysogorskiy, Jutta Rogal, and Ralf Drautz.
"Automated Free Energy Calculation from Atomistic Simulations." Physical Review Materials 5(10), 2021
DOI: 10.1103/PhysRevMaterials.5.103801

For more information contact:
sarath.menon@ruhr-uni-bochum.de
"""
import os
import numpy as np
import shutil
import argparse as ap
import subprocess
import yaml
import time
import datetime

from calphy.input import read_inputfile
from calphy.phase_diagram import prepare_inputs_for_phase_diagram


def phase_diagram():
    arg = ap.ArgumentParser()
    arg.add_argument("-i", "--input", required=True, type=str,
    help="name of the input file")
    args = vars(arg.parse_args())
    prepare_inputs_for_phase_diagram(args['input'])
