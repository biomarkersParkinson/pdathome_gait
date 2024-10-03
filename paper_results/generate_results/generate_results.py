import json
import logging
import os
import pandas as pd
import sys

from multiprocessing import Pool
from pdathome.evaluation import generate_results

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

def wrapper_function(subject_step):
    subject, step = subject_step
    return generate_results(subject, step)

def parallelize_function(n_processes, l_ids, step):
    with Pool(int(n_processes)) as p:
        try:
            # Prepare arguments as tuples
            subject_steps = [(subject, step) for subject in l_ids]
            p.map(wrapper_function, subject_steps)
            return 
        except Exception:
            logging.exception("Exception occurred during parallel processing.")
            return 

if __name__ == '__main__':
    nproc = int(sys.argv[1])  # Number of parallel processes
    l_ids = sys.argv[2:]      # List of ids to process

    l_steps = ['gait']

    for step in l_steps:
        parallelize_function(nproc, l_ids, step)