# -*- coding: utf-8 -*-
"""
fm2p/utils/multiprocessing_helpers.py

Shared-memory helpers for multiprocessing worker pools.

Functions
---------
init_worker
    Initialize a pool worker by attaching to shared memory buffers.


DMM, November 2025
"""

import numpy as np
from multiprocessing import Pool, cpu_count, shared_memory


def init_worker(shared_specs, params):
    """ Initialize a pool worker: attach to shared memory buffers and store params.

    Called as the pool initializer so each worker process has direct access to
    large arrays without pickling them per task.

    Parameters
    ----------
    shared_specs : dict
        Maps name to (shm_name, shape, dtype) tuples describing each shared array.
    params : dict
        Arbitrary parameter dict stored as a global for worker tasks to read.
    """

    global _shared_arrays, _params
    _shared_arrays = {}
    _params = params

    for name, (shm_name, shape, dtype) in shared_specs.items():
        shm = shared_memory.SharedMemory(name=shm_name)
        _shared_arrays[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
