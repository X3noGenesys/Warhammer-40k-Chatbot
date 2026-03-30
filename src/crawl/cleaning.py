from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import unicodedata
import json
import os

def load_jsonl(path):
    print("📚 Loading data...", end=" ")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} documents")
    return records

def clean_text(text):
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    patterns = [
        r"Cookies? policy.*", 
        r"Privacy policy.*", 
        r"©.*", 
        r"All rights reserved.*", 
        r"Advertisement", 
        r"- Pour plus de détails, voir l’article dédié.*", 
        r"^[| ]+", 
        r"\[[^\]]{1,30}\]" 
    ]

    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)

    return text.strip()


def text_formatting(records):
    print("🧹 Starting text formatting...", end=" ")
    for r in records:
        r["text"] = clean_text(r.get("text", ""))
    print("Text formatting completed")
    return records


def remove_exact_duplicates(records):
    print("🧹 Removing exact duplicates...", end=" ")
    seen = set()
    unique = []

    for r in records:
        key = hash(r["text"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"Deleted {len(records) - len(unique)} exact duplicates")
    return unique


def remove_similar_texts(records, similarity_threshold=0.95):
    print("🧹 Removing very similar texts...", end=" ")
    texts = [r["text"] for r in records]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    to_remove = set()

    for i in range(len(records)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(records)):
            if similarity_matrix[i, j] >= similarity_threshold:
                to_remove.add(j)

    cleaned = [
        r for idx, r in enumerate(records)
        if idx not in to_remove
    ]

    print(f"Deleted {len(records) - len(cleaned)} very similar texts")
    return cleaned


def quality_filter(records, min_length=300):
    print("🧹 Applying quality filter...", end=" ")
    filtered = [
        r for r in records
        if len(r["text"]) >= min_length
    ]

    print(f"Deleted {len(records) - len(filtered)} texts too short")
    return filtered

def text_cutoff(records, keyword):
    for r in records:
        texte = r.get("text", "")

        if keyword in texte:
            parties = texte.rsplit(keyword, 1)
            r["text"] = parties[0]

    return records

def clean_titles(records):
    print("🧹 Cleaning titles...", end=" ")
    suffix = " — Omnis Bibliotheca"
    prefix = "Catégorie:"
    n = 0

    for r in records:
        modified = False
        title = r.get("title", "")
        if title.endswith(suffix):
            title = title.removesuffix(suffix)
            modified = True
        if title.startswith(prefix):
            title = title.removeprefix(prefix)
            modified = True
        
        if modified:
            r["title"] = title
            n += 1
            

    print(f"Cleaned {n / len(records) * 100:.2f}% of titles")
    return records

def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            output = {
                "url": r["url"],
                "title": r["title"],
                "text": r["text"]
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_data_path = "data/crawled_data.jsonl"
    output_data_path = "data/cleaned_data.jsonl"

    os.makedirs("data", exist_ok=True)
    os.remove(output_data_path) if os.path.exists(output_data_path) else None
    open(output_data_path, "w").close()


    records = load_jsonl("data/crawled_data.jsonl")

    records = text_formatting(records)
    records = remove_exact_duplicates(records)
    records = remove_similar_texts(records, similarity_threshold=0.95)
    records = text_cutoff(records, "Médias Externes")
    records = text_cutoff(records, "Sources")
    records = text_cutoff(records, "Source")
    records = quality_filter(records, min_length=300)
    records = clean_titles(records)

    save_jsonl(records, output_data_path)
    print(f"✅ Cleaning completed → {output_data_path}")