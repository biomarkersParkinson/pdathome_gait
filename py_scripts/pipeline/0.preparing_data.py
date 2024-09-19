import functools
import sys

from multiprocessing import Pool

from pdathome.preprocessing import prepare_data
from pdathome.utils import PrintException


def pooling_func(subject):
    prepare_data(subject)

if __name__ == '__main__':
    nproc = sys.argv[1] # Number of parallel processes
    l_ids = sys.argv[2:] # List of ids to process

    with Pool(int(nproc)) as p:
        partial_process_file = functools.partial(pooling_func)
        try:
            p.map(partial_process_file, l_ids)     
        except:
            PrintException()

    

