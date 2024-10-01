import json
import logging
import sys
from multiprocessing import Pool
from pdathome.evaluation import generate_results

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

def parallelize_function(n_processes, l_ids, func, step):
    with Pool(int(n_processes)) as p:
        try:
            results = p.map(lambda subject: func(subject, step), l_ids)     
            return results
        except Exception:
            logging.exception("Exception occurred during parallel processing.")
            return None


if __name__ == '__main__':
    nproc = int(sys.argv[1])  # Number of parallel processes
    l_ids = sys.argv[2:]      # List of ids to process

    l_steps = ['gait', 'arm_activity', 'quantification']
    combined_results = {}

    for step in l_steps:
        results = parallelize_function(nproc, l_ids, generate_results, step)
        if results is not None:
            combined_results[step] = dict(zip(l_ids, results))

    # Store the combined results in a JSON file
    with open('combined_results.json', 'w') as f:
        json.dump(combined_results, f, indent=4)
