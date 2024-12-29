from collections import deque
from search import BFS, GreedyBFS, AStar
import time

START = "Coca-Cola"
TARGET = "Bowling"

def runner():
    searcher = AStar()

    start_t = time.time()
    node = searcher.find_path(START, TARGET)

    path = deque()
    while node:
        path.appendleft(node.state)
        node = node.parent

    print(f"\nFound {len(path)} Step Solution in {time.time() - start_t} seconds:")
    print(" -> ".join(path))

if __name__ == "__main__":
    runner()