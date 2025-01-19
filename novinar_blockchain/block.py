import hashlib
import json

class Block:
    def __init__(self, index, data, timestamp, prev_hash, difficulty, nonce, current_hash):
        self.index = index
        self.data = data
        self.timestamp = timestamp
        self.prev_hash = prev_hash
        self.difficulty = difficulty
        self.nonce = nonce
        self.current_hash = current_hash

    def calculate_hash(self):
        content = (
            str(self.index)
            + json.dumps(self.data, sort_keys=True)
            + str(self.timestamp)
            + self.prev_hash
            + str(self.difficulty)
            + str(self.nonce)
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

