import logging
import sys

from multiprocessing import Pool
from pdathome.classification import detect_gait, filter_gait
from pdathome.preprocessing import prepare_data, preprocess_gait, preprocess_filtering_gait

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Mapping of step numbers to functions
steps_map = {
    '1': prepare_data,
    '2': preprocess_gait,
    '3': detect_gait,
    '4': preprocess_filtering_gait,
    '5': filter_gait,
}


def parallelize_function(n_processes, l_ids, func):
    with Pool(int(n_processes)) as p:
        try:
            p.map(func, l_ids)     
        except Exception:
            logging.exception("Exception occurred during parallel processing.")


if __name__ == '__main__':
    nproc = int(sys.argv[1]) # Number of parallel processes
    steps = sys.argv[2] # Steps to run: (1) prepare_data, (2) preprocess_gait, (3) detect_gait, (4) preprocess_filtering_gait, (5) filter_gait (e.g., 345)
    l_ids = sys.argv[3:] # List of ids to process

    # Run the specified steps
    for step in steps:
        func = steps_map.get(step)
        if func:
            parallelize_function(nproc, l_ids, func)
        else:
            logging.warning(f"Step {step} is not recognized.")