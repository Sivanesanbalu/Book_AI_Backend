from collections import deque, Counter
import time

# how many recent frames to remember
WINDOW_SIZE = 8

# how long same result must stay (seconds)
STABLE_TIME = 0.7


class TitleMemory:

    def __init__(self):
        self.history = deque(maxlen=WINDOW_SIZE)
        self.last_stable = None
        self.last_change_time = time.time()

    def update(self, title: str):

        now = time.time()
        self.history.append(title)

        # count votes
        counts = Counter(self.history)
        best_title, votes = counts.most_common(1)[0]

        confidence = votes / len(self.history)

        # require majority vote
        if confidence >= 0.6:

            if best_title != self.last_stable:
                self.last_change_time = now
                self.last_stable = best_title
                return None  # wait for stability

            # stable long enough
            if now - self.last_change_time >= STABLE_TIME:
                return best_title

        return None


# global memory per user
user_memories = {}

def get_memory(uid: str):
    if uid not in user_memories:
        user_memories[uid] = TitleMemory()
    return user_memories[uid]