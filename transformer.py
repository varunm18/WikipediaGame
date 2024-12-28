from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, SimilarityFunction, CrossEncoder
import numpy as np
import requests
import wikipedia, faiss
import time
from urllib.parse import quote

START = "Confectionery store"
TARGET = "costco"

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

        existing_links.extend(page["title"] for page in pages.values() if "pageid" in page and page["pageid"] >= 0 )
        
        # Check for 'continue' to fetch the next batch
        if "continue" in response:
            params.update(response["continue"])
        else:
            break

    return existing_links 

def get_category_keywords(page_title):
    try:
        page = wikipedia.page(page_title)
        return ' '.join(page.categories)
    except:
        return ''

links = get_all_internal_links(START)

# # Load a pretrained Sentence Transformer model
# model = SentenceTransformer("multi-qa-MiniLM-L6-dot-v2")

# embeddings = model.encode(TARGET)
# print(embeddings)

# similarities = model.similarity(embeddings, embeddings)
# print(similarities)


# CROSS TRAINER
# model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

# # 2. We rank all sentences in the corpus for the query
# ranks = model.rank(TARGET, links)

# # Print the scores
# print("Query: ", TARGET)
# with open("ranks.txt", "w") as f:
#     for rank in ranks:
#         f.write(f"{rank['score']:.2f}\t{links[rank['corpus_id']]}\n")

# SENTENCE TRANSFORMER
model = SentenceTransformer('BAAI/bge-base-en-v1.5')
# jake = wikipedia.page("Jake Elliott (American football)")
# Generate embeddings
# target_embedding = model.encode(f"{wikipedia.summary(TARGET)}")
# print(target_embedding.shape)
# link_embeddings = model.encode(links)
# print(link_embeddings.shape)
# Calculate cosine similarities
# similarities = [1 - cosine(target_embedding, emb) for emb in link_embeddings]
# most_similar_indices = np.argsort(similarities)[::-1]
# print([(links[i], similarities[i]) for i in most_similar_indices])

# # Sort links by similarity score
# ranked_links = [(links[i], float(similarities[i])) 
#                 for i in range(len(links))]
# ranked_links.sort(key=lambda x: x[1], reverse=True)

# with open("ranks.txt", "w") as f:
#     for rank in ranked_links:
#         f.write(f"{rank[1]:.2f}\t{rank[0]}\n") 

start_t = time.time()
top_n = int(len(links))

goal_embedding = model.encode(wikipedia.summary(TARGET, auto_suggest=False))
summaries = []
for link in links:
    try:
        summaries.append(link, auto_suggest=False)
    except:
        summaries.append(link)
link_embeddings = model.encode(summaries)

goal_embedding = goal_embedding / np.linalg.norm(goal_embedding)
link_embeddings = link_embeddings / np.linalg.norm(link_embeddings, axis=1, keepdims=True)

# similarities = np.dot(link_embeddings, goal_embedding)
# link_similarity_pairs = list(zip(links, similarities))

# # # Sort the list by similarity in descending order
# sorted_links = sorted(link_similarity_pairs, key=lambda x: x[1], reverse=True)

dimension = link_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

index.add(link_embeddings)
distances, top_indices = index.search(goal_embedding.reshape(1, -1), 100)
top_links = [links[i] for i in top_indices[0]]
# with open("ranks.txt", "w") as f:
#     for link, similarity in sorted_links[:top_n]:  # Use `top_n` if you want the top results
#         f.write(f"{link}: {similarity}\n")

print(time.time() - start_t)