"""
xMIGCS Package
Humanoid robot control system package
"""
import pathlib
from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)

XMIGCS_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent  # .../xmigcs

# import os


# def _configure_default_thread_limits():
#     """Keep third-party native thread pools small by default."""
#     thread_env_defaults = {
#         "OMP_NUM_THREADS": "1",
#         "OPENBLAS_NUM_THREADS": "1",
#         "MKL_NUM_THREADS": "1",
#         "NUMEXPR_NUM_THREADS": "1",
#         "BLIS_NUM_THREADS": "1",
#         "VECLIB_MAXIMUM_THREADS": "1",
#     }
#     for env_name, value in thread_env_defaults.items():
#         os.environ.setdefault(env_name, value)


# _configure_default_thread_limits()

# if __name__ == "__main__":
#     logger.debug(XMIGCS_ROOT_DIR)
