import logging
import sys

from functools import partial
from multiprocessing import Pool

from pdathome.constants import global_constants as gc
from pdathome.classification import train_test_gait_detection, train_test_filtering_gait, store_gait_detection, store_filtering_gait
from pdathome.preprocessing import prepare_data, preprocess_gait_detection, preprocess_filtering_gait

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Mapping of step numbers to functions
steps_map = {
    '1': prepare_data,
    '2': preprocess_gait_detection,
    '3': train_test_gait_detection,
    '4': store_gait_detection,
    '5': preprocess_filtering_gait,
    '6': train_test_filtering_gait,
    '7': store_filtering_gait,
}

# Define which steps should run in parallel
parallel_steps = {'1', '2', '3', '5', '6'}  

def parallelize_function(n_processes, l_ids, func):
    with Pool(int(n_processes)) as p:
        try:
            p.map(func, l_ids)     
        except Exception:
            logging.exception("Exception occurred during parallel processing.")

def sequential_function(func):
    try:
        func()
    except Exception:
        logging.exception(f"Exception occurred during sequential processing.")

if __name__ == '__main__':
    nproc = int(sys.argv[1])  # Number of parallel processes
    steps = sys.argv[2]       # Steps to run: e.g., 345
    l_ids = sys.argv[3:]      # List of ids to process

    # Choose which classifiers to run
    gd_classifiers = [gc.classifiers.LOGISTIC_REGRESSION, gc.classifiers.RANDOM_FOREST]
    fg_classifiers = [gc.classifiers.LOGISTIC_REGRESSION, gc.classifiers.RANDOM_FOREST]

    gd_gsearch = False
    fg_gsearch = False

    # No need for nested parallelization
    n_jobs = 1

    for step, func in steps_map.items():
        if step == '3':
            steps_map[step] = partial(func, l_classifiers=gd_classifiers, gsearch=gd_gsearch, n_jobs=n_jobs)
        elif step == '4':
            steps_map[step] = partial(func, l_classifiers=gd_classifiers)
        elif step == '6':
            steps_map[step] = partial(func, l_classifiers=fg_classifiers, gsearch=fg_gsearch, n_jobs=n_jobs)
        elif step == '7': 
            steps_map[step] = partial(func, l_classifiers=fg_classifiers)

    # Run the specified steps
    for step in steps:
        func = steps_map.get(step)
        if func:
            if step in parallel_steps:
                parallelize_function(nproc, l_ids, func)
            else:
                sequential_function(func)
        else:
            logging.warning(f"Step {step} is not recognized.")
