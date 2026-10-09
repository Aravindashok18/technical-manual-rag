import os

# torch and faiss each ship their own OpenMP runtime. On macOS, running both multi-threaded in one
# process crashes with a segmentation fault (or "OMP: Error #15"), so pin OpenMP to one thread
# before either library is imported. Query-time cost is negligible for an index this size.
os.environ.setdefault("OMP_NUM_THREADS", "1")
