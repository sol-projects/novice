import time
from block import Block

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.block_time = 10
        self.adjustment_interval = 10

    def create_genesis_block(self):
        genesis = Block(
            index=0,
            data="Genesis Block",
            timestamp=int(time.time()),
            prev_hash="0",
            difficulty=5,
            nonce=0,
            current_hash=""
        )
        genesis.current_hash = genesis.calculate_hash()
        return genesis

    def get_latest_block(self):
        return self.chain[-1]

    def is_valid_new_block(self, new_block, prev_block):
        if new_block.index != prev_block.index + 1:
            return False
        if new_block.prev_hash != prev_block.current_hash:
            return False
        if new_block.calculate_hash() != new_block.current_hash:
            return False
        if not new_block.current_hash.startswith("0" * new_block.difficulty):
            return False
        if new_block.timestamp - time.time() > 60:
            return False
        if new_block.timestamp < prev_block.timestamp - 60:
            return False
        return True

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            if not self.is_valid_new_block(self.chain[i], self.chain[i-1]):
                return False
        return True

    def get_difficulty(self):
        latest = self.get_latest_block()
        if latest.index % self.adjustment_interval == 0 and latest.index != 0:
            return self.adjust_difficulty()
        return latest.difficulty

    def adjust_difficulty(self):
        latest = self.get_latest_block()
        idx = max(0, latest.index - self.adjustment_interval)
        old = self.chain[idx]
        expected_time = self.block_time * self.adjustment_interval
        actual_time = latest.timestamp - old.timestamp
        if actual_time < expected_time // 2:
            return old.difficulty + 1
        elif actual_time > expected_time * 2:
            return max(1, old.difficulty - 1)
        return old.difficulty

    def add_block(self, new_block):
        prev_block = self.get_latest_block()
        if self.is_valid_new_block(new_block, prev_block):
            self.chain.append(new_block)
            return True
        return False

