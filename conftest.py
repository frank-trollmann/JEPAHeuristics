import os

# torch and xgboost each bundle their own copy of the OpenMP runtime
# (libomp) on macOS. Loading both in the same process can segfault unless
# this is set - must happen before torch/xgboost get imported anywhere,
# which is why this lives in a root conftest.py (pytest loads it first).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
