from mpi4py import MPI
from parallel_blockchain import ParallelBlockchain
import argparse
import os
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parallel blockchain miner")
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Number of threads per node. Use a negative number to divide the number of available cores by the absolute value (e.g., -2 means half the cores)."
    )
    return parser.parse_args()

def determine_thread_count(threads_arg):
    num_cores = os.cpu_count()
    if threads_arg < 0:
        return max(1, num_cores // abs(threads_arg))
    return max(1, threads_arg)

def main():
    args = parse_arguments()
    thread_count = determine_thread_count(args.threads)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"Rank {rank}/{size} using {thread_count} threads for mining.")
    pb = ParallelBlockchain(thread_count=thread_count)

    while True:
        data_to_mine = None
        if rank == 0:
            user_input = input("\nEnter data to mine (or 'quit' to exit): ").strip()
            if user_input.lower() == "quit":
                data_to_mine = "QUIT_NOW"
            else:
                data_to_mine = user_input

        data_to_mine = comm.bcast(data_to_mine, root=0)

        if data_to_mine == "QUIT_NOW":
            if rank == 0:
                print("Exiting...")
            break

        start_time = time.time()
        success = pb.parallel_mine_block(data_to_mine)
        end_time = time.time()

        if rank == 0:
            duration = end_time - start_time
            if success:
                print(f"Block mined successfully with data: '{data_to_mine}' in {duration:.6f} seconds")
            else:
                print("No block was mined (unexpected).")
            print("Current chain:")
            for block in pb.chain:
                print(vars(block))

if __name__ == "__main__":
    main()

