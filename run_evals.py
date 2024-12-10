import subprocess
import os

# Generate all combinations of instance and test indices
tasks = [(i, j) for i in range(32) for j in range(5)]

# Maximum number of parallel processes
MAX_PARALLEL_PROCESSES = os.cpu_count()  # Use the number of CPU cores

def run_in_parallel(tasks):
    """
    Executes tasks in parallel using UNIX commands.
    """
    processes = []

    for instance_index, test_index in tasks:
        cmd = f"python evaluate_instance.py {instance_index} {test_index}"
        print(f"Starting: {cmd}")
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

        # If we reach the maximum number of parallel processes, wait for some to finish
        if len(processes) >= MAX_PARALLEL_PROCESSES:
            for p in processes:
                p.wait()
            processes = []  # Clear the list after all processes are done

    # Wait for any remaining processes to finish
    for p in processes:
        p.wait()

if __name__ == "__main__":
    run_in_parallel(tasks)
    print("All tasks completed.")
