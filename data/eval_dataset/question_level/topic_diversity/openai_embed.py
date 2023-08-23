"""
Adapted From openchat: ttps://github.com/imoneoi/openchat
- Embed text documents using OpenAI embedding API.
Usage:
python -m openai_embed --in-file ./10k/allq/s.json --out-file ./10k/allq/s_embedded.json
"""
import json
import argparse

import openai
import tiktoken
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm


MAX_TOKENS = 2048 
BATCH_SIZE = 16 
MODEL_TYPE = "text-embedding-ada-002"
MODEL_TOKENIZER = tiktoken.encoding_for_model(MODEL_TYPE) 
# set your own api key.
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(6))
def embedding_with_backoff(**kwargs):
    return openai.Embedding.create(**kwargs)

def preprocess_text(text: str): 
    tokens = MODEL_TOKENIZER.encode(text, disallowed_special=()) 
    tokens = tokens[:MAX_TOKENS]
    return tokens

def calculate_embeddings(samples): 
    embeddings = []

    for start_idx in tqdm(range(0, len(samples), BATCH_SIZE)):
        sample_chunk = samples[start_idx: start_idx + BATCH_SIZE]
        tokens_chunk = list(map(preprocess_text, sample_chunk))
    
        response = embedding_with_backoff(model=MODEL_TYPE, input=tokens_chunk)
        
        for i, be in enumerate(response["data"]):
            assert i == be["index"] 

        embeddings_chunk = [e["embedding"] for e in response["data"]]
        embeddings.extend(embeddings_chunk)
        
    return embeddings
   
def main(args):
    samples = json.load(open(args["in_file"], "r", encoding='utf-8')) 
    embeddings = calculate_embeddings(samples)

    json.dump(embeddings, open(args["out_file"], "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str)
    parser.add_argument("--out-file", type=str)
    args = parser.parse_args()

    main(vars(args))
