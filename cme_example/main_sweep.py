# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from functools import partial

import numpy as np
import jax
from tqdm.contrib.concurrent import process_map

from actuator_constants import files

from variation_relax import run_relaxation
from variation_run import run_parameter_variation
from variation_view import run_plot

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":

    f_run = partial(run_relaxation, n_vertices=1000)
    r = process_map(f_run, files, max_workers=12)

    f_run = partial(run_parameter_variation, _Ksg_=np.linspace(0, 2, 1 + 2**6))
    r = process_map(f_run, files, max_workers=12)

    r = process_map(run_plot, files, max_workers=12)
