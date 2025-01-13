import time
import multiprocessing
from block import Block
from blockchain import Blockchain


class ParallelBlockchain(Blockchain):
    def __init__(self, thread_count=2):
        super().__init__()
        self.num_processes = thread_count 
        self.found_nonce = multiprocessing.Value('i', -1)  
        self.solution_hash = multiprocessing.Array('c', 64) 

    def parallel_mine_block(self, data):
        prev_block = self.get_latest_block()
        index = prev_block.index + 1
        difficulty = self.get_difficulty()
        timestamp = int(time.time())

        new_block_template = Block(
            index=index,
            data=data,
            timestamp=timestamp,
            prev_hash=prev_block.current_hash,
            difficulty=difficulty,
            nonce=0,
            current_hash=""
        )

        def mine_nonce_range(start_nonce, step, found_nonce, solution_hash):
            local_nonce = start_nonce
            while found_nonce.value == -1:
                new_block = Block(
                    index=new_block_template.index,
                    data=new_block_template.data,
                    timestamp=new_block_template.timestamp,
                    prev_hash=new_block_template.prev_hash,
                    difficulty=new_block_template.difficulty,
                    nonce=local_nonce,
                    current_hash=""
                )
                h = new_block.calculate_hash()
                if h.startswith("0" * difficulty):
                    with found_nonce.get_lock():
                        if found_nonce.value == -1: 
                            found_nonce.value = local_nonce
                            solution_hash.value = h.encode('utf-8')
                    break
                local_nonce += step

        processes = []
        for p_id in range(self.num_processes):
            process = multiprocessing.Process(
                target=mine_nonce_range,
                args=(p_id, self.num_processes, self.found_nonce, self.solution_hash)
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        if self.found_nonce.value != -1:
            mined_block = Block(
                index=new_block_template.index,
                data=new_block_template.data,
                timestamp=new_block_template.timestamp,
                prev_hash=new_block_template.prev_hash,
                difficulty=new_block_template.difficulty,
                nonce=self.found_nonce.value,
                current_hash=self.solution_hash.value.decode('utf-8')
            )
            self.add_block(mined_block)
            return True
        return False

