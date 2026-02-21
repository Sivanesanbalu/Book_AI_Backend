import time
from collections import deque
from rapidfuzz import fuzz

STABLE_FRAMES = 4
TIMEOUT = 25  # seconds


# -------------------------------------------------------
# FUZZY SAME TITLE CHECK  ⭐ VERY IMPORTANT
# -------------------------------------------------------
def similar(a, b):
    return fuzz.token_set_ratio(a, b) >= 82


class TitleMemory:

    def __init__(self):
        self.buffer = deque(maxlen=STABLE_FRAMES)
        self.last_update = 0
        self.locked_title = None

    # called continuously while scanning
    def update(self, title: str):

        if not title:
            return None

        now = time.time()

        # reset if camera paused / moved away
        if now - self.last_update > TIMEOUT:
            self.buffer.clear()
            self.locked_title = None

        self.last_update = now
        self.buffer.append(title)

        if len(self.buffer) < STABLE_FRAMES:
            return None

        # --------------------------------------------------
        # FUZZY STABILITY CHECK (instead of exact match)
        # --------------------------------------------------
        base = self.buffer[0]
        stable_count = 0

        for t in self.buffer:
            if similar(base, t):
                stable_count += 1

        # require majority agreement
        if stable_count >= STABLE_FRAMES - 1:
            self.locked_title = base
            return base

        return None

    # called when user presses capture
    def confirm(self):
        return self.locked_title


# memory per user
_user_memory = {}

def get_memory(uid: str):
    if uid not in _user_memory:
        _user_memory[uid] = TitleMemory()
    return _user_memory[uid]