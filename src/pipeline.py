import os
from dotenv import load_dotenv
from tqdm import tqdm
import json

from ingest import load_articles
from embeddings import get_embedding
from retriever import Retriever
from extractor import extract_info
from validator import validate_output

load_dotenv()

def run_pipeline():
    print("Loading articles...")
    articles = load_articles()

    print("Generating embeddings...")
    embeddings = [get_embedding(a) for a in articles]

    retriever = Retriever(embeddings, articles)

    results = []

    print("Processing articles...")
    for article in tqdm(articles):
        query_emb = get_embedding(article)
        context = retriever.retrieve(query_emb)

        extracted = extract_info(article, context)

        is_valid = validate_output(extracted, article)

        result = {
            "input": article,
            "output": extracted,
            "validated": is_valid
        }

        results.append(result)

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Done! Results saved in outputs/results.json")


if __name__ == "__main__":
    run_pipeline()
