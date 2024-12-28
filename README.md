# Wikipedia Game
Search algorithms that find a path between two Wikipedia pages through links on the pages

## Description:
#### Utility Functions-
* Gets all internal links on a given page from [WikiMedia API](https://www.mediawiki.org/wiki/API:Main_page)
#### BFS-
* Uses Doubly Linked List as queue (deque) for frontier
* Searches by Depth Level
#### Greedy BFS-
* Uses Heap as priority queue (heapq) for frontier
* Searches by heuristic function using [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) Transformer model through [SentenceTransformer](https://sbert.net) module
* Finds Cosine Similarity 
