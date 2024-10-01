import logging
import sys

from multiprocessing import Pool

from pdathome.evaluation import generate_results

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

def parallelize_function(n_processes, l_ids, func):
    with Pool(int(n_processes)) as p:
        try:
            p.map(func, l_ids)     
        except Exception:
            logging.exception("Exception occurred during parallel processing.")


if __name__ == '__main__':
    nproc = int(sys.argv[1])  # Number of parallel processes
    l_ids = sys.argv[2:]      # List of ids to process

    parallelize_function(nproc, l_ids, generate_results)
