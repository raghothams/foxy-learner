# -*- coding: utf-8 -*-

import sys

from sklearn.datasets import load_svmlight_file
from sklearn.externals.joblib import Memory, Parallel, delayed
from tempfile import mkdtemp


# function to read train data
# this function call has to be memory mapped by parent 
#   so that children do not create copies
def get_train_data(file_path):
    print "will read now"
    X,y = load_svmlight_file(file_path, multilabel=True)


# dummy function to call memory mapped function to read training data
def child_read_train_data(mem_cached_func, file_path):
    mem_cached_func(file_path)

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print "Usage: \nmemmap <file-path>\n"
        exit("No file path specified")

    file_path = sys.argv[1]
    print "file to be read ",file_path

# get a temp directory to cache the array
    cache_dir = mkdtemp()

# get Memory instance
    mem = Memory(cachedir=cache_dir, mmap_mode='r', verbose=5)
    memmed_getter = mem.cache(get_train_data)

    Parallel(n_jobs=3, verbose=3) (delayed(child_read_train_data)\
            (memmed_getter, file_path) for _ in range(5))

    del memmed_getter
    del mem
    del cache_dir

