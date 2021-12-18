import importlib
import os
import platform
from functools import wraps
from time import time

from loguru import logger
from torch import multiprocessing


def get_exe_time(func):
    def inner_fn(*args, **kwargs):
        t0 = time()
        result = func(*args, **kwargs)
        t1 = time()
        return (t1 - t0), result

    return inner_fn


def print_exe_time(log_level="info"):
    def inner_fn(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time()
            result = func(*args, **kwargs)
            t1 = time()
            logger.log(log_level.upper(),
                       f"{func.__name__} executed in {t1 - t0: 0.3f} s")
            return result

        return wrapper

    return inner_fn


def is_installed(module_name: str):
    return importlib.util.find_spec(module_name) is not None


def setup_os_platform():
    if platform.system() == "Darwin":
        logger.info(f"Platform: {platform.system()}")
        # Python 3.8 changed to 'spawn' but that doesn't work with PyTorch DataLoader w n_workers>0
        multiprocessing.set_start_method("fork", force=True)
        # workaround "OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    else:
        multiprocessing.set_start_method("spawn", force=True)
        logger.info(f"Platform: {platform.system()}")
