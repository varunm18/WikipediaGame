import requests
from collections import deque

START = "Albert Einstein"
TARGET = "Television"

class Node():
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent
    
class QueueFrontier():
    def __init__(self):
        self.frontier = deque()

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            return self.frontier.popleft()

def get_all_internal_links(title):
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "generator": "links",
        "gpllimit": "max",
        "gplnamespace": "0",
        "format": "json"
    }
    
    existing_links = []
    while True:
        response = requests.get(base_url, params=params).json()

        pages = response.get("query", {}).get("pages", {})
        existing_links.extend(page["title"] for page in pages.values())
        
        # Check for 'continue' to fetch the next batch
        if "continue" in response:
            params.update(response["continue"])
        else:
            break
    
    return existing_links

def shorthest_path(source, target):
    frontier = QueueFrontier()
    frontier.add(Node(source, None))
    explored = set()

    while True:
        if frontier.empty():
            return None
        
        node = frontier.remove()
        explored.add(node.state)
        print(f"Checking {node.state}")

        if node.state == target:
            return node

        for link in get_all_internal_links(node.state):
            if link == target:
                return Node(link, node)

            if link not in explored and not frontier.contains_state(link):
                frontier.add(Node(link, node))

def runner():
    node = shorthest_path(START, TARGET)
    path = deque()
    while node:
        path.appendleft(node.state)
        node = node.parent
    print(path)

if __name__ == "__main__":
    runner()