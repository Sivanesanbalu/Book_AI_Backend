import time
from collections import deque

STABLE_FRAMES = 3
TIMEOUT = 25  # seconds

class TitleMemory:

    def __init__(self):
        self.buffer = deque(maxlen=STABLE_FRAMES)
        self.last_update = 0
        self.locked_title = None

    # called during scan
    def update(self, title: str):

        now = time.time()

        # reset if user moved camera
        if now - self.last_update > TIMEOUT:
            self.buffer.clear()
            self.locked_title = None

        self.last_update = now
        self.buffer.append(title)

        if len(self.buffer) < STABLE_FRAMES:
            return None

        # stable detection
        if len(set(self.buffer)) == 1:
            self.locked_title = title
            return title

        return None

    # called during capture
    def confirm(self):
        return self.locked_title


# memory per user
_user_memory = {}

def get_memory(uid: str):
    if uid not in _user_memory:
        _user_memory[uid] = TitleMemory()
    return _user_memory[uid]