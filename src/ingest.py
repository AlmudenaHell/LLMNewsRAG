import os

def load_articles(data_path="data"):
    articles = []
    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            with open(os.path.join(data_path, file), "r") as f:
                articles.append(f.read())
    return articles
