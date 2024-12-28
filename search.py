import requests
from collections import deque
import heapq
from urllib.parse import quote
from sentence_transformers import SentenceTransformer, SimilarityFunction, CrossEncoder
import wikipedia
import numpy as np, faiss
    
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

class PriorityQueueFrontier():
    def __init__(self):
        # Heap
        self.frontier = []

    def add(self, node):
        heapq.heappush(self.frontier, node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            return heapq.heappop(self.frontier)

class Search():
    class Node():
        def __init__(self, state, parent):
            self.state = state
            self.parent = parent

    def get_all_internal_links(self, title):
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title.replace(" ", "_"),
            "generator": "links",
            "gpllimit": "max",
            "gplnamespace": "0",
            "format": "json"
        }
        existing_links = []
        while True:
            response = requests.get(base_url, params=params).json()
        
            pages = response.get("query", {}).get("pages", {})

            existing_links.extend(page["title"] for page in pages.values() if "pageid" in page and page["pageid"] >= 0 )
            
            # Check for 'continue' to fetch the next batch
            if "continue" in response:
                params.update(response["continue"])
            else:
                break

        return existing_links   
    
    def print_node(self, node):
        print(f"Checking {node.state}", end="")

        curr = node.parent
        while curr:
            print(f" FROM {curr.state}", end="")
            curr = curr.parent
            
        print()

class BFS(Search):
    def __init__(self):
        self.frontier = QueueFrontier()
        self.explored = set()

    def find_path(self, source, target):
        self.frontier.add(self.Node(source, None))

        while True:
            if self.frontier.empty():
                return None
            
            node = self.frontier.remove()
            self.explored.add(node.state)

            self.print_node(node)

            if node.state == target:
                return node

            for link in self.get_all_internal_links(node.state):
                if link == target:
                    return self.Node(link, node)

                if link not in self.explored and not self.frontier.contains_state(link):
                    self.frontier.add(self.Node(link, node))

class GreedyBFS(Search):
    class Node(Search.Node):
        def __init__(self, state, parent, similarity=None):
            super().__init__(state, parent)
            self.similarity = similarity
        
        def __lt__(self, other):
            return self.similarity > other.similarity
        
    def __init__(self, model='BAAI/bge-base-en-v1.5'):
        self.frontier = PriorityQueueFrontier()
        self.explored = set()
        self.model = SentenceTransformer(model)
        
    def get_heuristics(self, links, top_n=None):
        if top_n is None:
            top_n = len(links)

        summaries = []
        for link in links:
            try:
                summaries.append(link, auto_suggest=False)
            except:
                summaries.append(link)

        link_embeddings = self.model.encode(summaries)
        link_embeddings = link_embeddings / np.linalg.norm(link_embeddings, axis=1, keepdims=True)

        # Cosine Similarities, Already Normalized Vectors
        # similarities = np.dot(link_embeddings, self.target_embedding)

        # return sorted(list(zip(similarities, links)), key=lambda x: x[0], reverse=True)

        index = faiss.IndexFlatIP(link_embeddings.shape[1])
        index.add(link_embeddings)

        similarities, top_indices = index.search(self.target_embedding.reshape(1, -1), top_n)

        return list(zip(similarities[0], [links[i] for i in top_indices[0]]))

    def find_path(self, source, target):
        self.frontier.add(self.Node(source.title(), None))

        self.target_embedding = self.model.encode(wikipedia.summary(target, auto_suggest=False))
        self.target_embedding /= np.linalg.norm(self.target_embedding)

        while True:
            if self.frontier.empty():
                return None
            
            node = self.frontier.remove()
            self.explored.add(node.state)

            self.print_node(node)

            if node.state.lower() == target.lower():
                return node
            
            links = self.get_heuristics(self.get_all_internal_links(node.state))

            for link in links:
                if link[1].lower() == target.lower():
                    return self.Node(link[1], node, link[0])

                if link[1] not in self.explored and not self.frontier.contains_state(link[1]):
                    self.frontier.add(self.Node(link[1], node, link[0]))

class AStar(GreedyBFS):
    class Node(Search.Node):
        COST_K = 0.03
        SIMILARITY_K = 1

        def __init__(self, state, parent, depth, similarity=None):
            super().__init__(state, parent)

            self.depth = depth
            self.similarity = similarity

            self.cost = (self.depth * self.COST_K) + ((1 - self.similarity) * self.SIMILARITY_K)
        
        def __lt__(self, other):
            return self.cost < other.cost
        
    def __init__(self, model='BAAI/bge-base-en-v1.5'):
        super().__init__(model)
    
    def find_path(self, source, target):
        self.frontier.add(self.Node(source.title(), None, 0, 1))

        self.target_embedding = self.model.encode(wikipedia.summary(target, auto_suggest=False))
        self.target_embedding /= np.linalg.norm(self.target_embedding)

        while True:
            if self.frontier.empty():
                return None
            
            node = self.frontier.remove()
            self.explored.add(node.state)

            self.print_node(node)

            if node.state.lower() == target.lower():
                return node
            
            links = self.get_heuristics(self.get_all_internal_links(node.state))

            for link in links:
                if link[1].lower() == target.lower():
                    return self.Node(link[1], node, node.depth + 1, link[0])

                if link[1] not in self.explored and not self.frontier.contains_state(link[1]):
                    self.frontier.add(self.Node(link[1], node, node.depth + 1, link[0]))
