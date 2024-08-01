import random
import collections

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, data):
        print('push buffer')
        self.buffer.append(data)

    def sample(self, batch_size):
        print('sample buffer')
        # print('self.buffer',self.buffer)
        return random.sample(self.buffer, batch_size)

    def clear(self):
        print('clear buffer')
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
    
    def is_full(self):
        print('check whether buffer is full')
        return len(self.buffer) >= self.capacity