import functools
import sys

from multiprocessing import Pool
from pdathome.classification import detect_gait, filter_gait
from pdathome.preprocessing import prepare_data, preprocess_gait, preprocess_filtering_gait
from pdathome.utils import PrintException


def parallelize_function(n_processes, l_ids, func):
    with Pool(int(n_processes)) as p:
        partial_process_file = functools.partial(func)
        try:
            p.map(partial_process_file, l_ids)     
        except:
            PrintException()


if __name__ == '__main__':
    nproc = sys.argv[1] # Number of parallel processes
    l_ids = sys.argv[2:] # List of ids to process

    parallelize_function(nproc, l_ids, prepare_data)
    parallelize_function(nproc, l_ids, preprocess_gait)
    parallelize_function(nproc, l_ids, detect_gait)
    parallelize_function(nproc, l_ids, preprocess_filtering_gait)
    parallelize_function(nproc, l_ids, filter_gait)



    

    

    

