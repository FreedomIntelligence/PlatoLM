import json
import time

import numpy as np
from scipy import spatial 


def read_json_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data

def compute_cosine_sims(embeddings):
    embeddings = np.array(embeddings)

    cosine_dist = spatial.distance.cdist(embeddings, embeddings, 'cosine')
    cosine_sims = (1 - cosine_dist)
    
    return cosine_sims

def compute_avg_cosine_sims(embedding_path: str):
    embeddings = read_json_file(embedding_path)
    
    start = time.time() 
    cosine_sims = compute_cosine_sims(embeddings)
    end = time.time() 
    
    avg_cosine_sims = np.mean(cosine_sims)
    print("Time elapsed:", end - start) 
    
    return avg_cosine_sims

if __name__ == "__main__":
    avg = compute_avg_cosine_sims('./s_embedded.json')
    print('avg: ', avg)
