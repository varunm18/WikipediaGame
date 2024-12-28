# Wikipedia Game
Search algorithms that find a path between two Wikipedia pages through links on the pages

## Description:
#### Utility Functions-
* Gets all internal links on a given page from [WikiMedia API](https://www.mediawiki.org/wiki/API:Main_page) using [requests](https://requests.readthedocs.io/en/latest/) module
#### BFS-
* Uses Doubly Linked List as queue (deque) for frontier
* Searches by Depth Level
#### Greedy BFS-
* Uses Heap as priority queue (heapq) for frontier
* Searches by heuristic function using [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) Transformer model from Hugging Face through [SentenceTransformer](https://sbert.net) module
  * Finds Cosine Similarity between Target Embedding and all Link Embeddings
* Searches by Highest Cosine Similarity
#### A*-
* Uses Heap as priority queue (heapq) for frontier
* Uses Inverse Cosine Similarity (Cosine Distance) through the same heurisitic as Greedy BFS with depth level
  * `self.cost = (self.depth * self.COST_K) + ((1 - self.similarity) * self.SIMILARITY_K)`
* Searches by Lowest Cost
