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

def wrapper_function(subject_step):
    subject, step = subject_step
    return generate_results(subject, step)

def parallelize_function(n_processes, l_ids, func, step):
    with Pool(int(n_processes)) as p:
        try:
            # Prepare arguments as tuples
            subject_steps = [(subject, step) for subject in l_ids]
            results = p.map(wrapper_function, subject_steps)
            return results
        except Exception:
            logging.exception("Exception occurred during parallel processing.")
            return None

if __name__ == '__main__':
    nproc = int(sys.argv[1])  # Number of parallel processes
    l_ids = sys.argv[2:]      # List of ids to process

    l_steps = ['gait', 'arm_activity', 'quantification']

    for step in l_steps:
        results = parallelize_function(nproc, l_ids, generate_results, step)
        if results is not None:
            # Create a dictionary to store results for the current step
            step_results = {id_: result for id_, result in zip(l_ids, results) if result is not None}

            # Store the results for the current step in a separate JSON file
            step_filename = f'results_{step}.json'
            with open(step_filename, 'w') as f:
                json.dump(step_results, f, indent=4)

            logging.info(f"Results for step '{step}' stored in '{step_filename}'.")
