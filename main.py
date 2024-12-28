from collections import deque
from search import BFS, GreedyBFS

START = "South Brunswick, New Jersey"
TARGET = "Denver Broncos"

def runner():
    searcher = GreedyBFS()
    node = searcher.shorthest_path(START, TARGET)
    path = deque()
    while node:
        path.appendleft(node.state)
        node = node.parent

    print("\nFound Solution:")
    print(" -> ".join(path))

if __name__ == "__main__":
    runner()