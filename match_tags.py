import json
from transformers import AutoModel, AutoTokenizer
import torch
from scipy.spatial.distance import cosine
import numpy as np

def load_model_tokenizer(model_name="dmis-lab/biobert-v1.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

def get_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    # Use mean pooling and squeeze to get a 1D representation
    embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze().detach().numpy()
    return embeddings


# Function to compute cosine similarity
def compute_similarity(embedding1, embedding2):
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()
    return 1 - cosine(embedding1, embedding2)

# Load model and tokenizer
model, tokenizer = load_model_tokenizer()

with open("vocabulary.json") as vocab_file:
    vocabulary = json.load(vocab_file)

with open("relevant_terms.json") as rel_terms_file:
    documents = json.load(rel_terms_file)
    

unique_tfidf_terms = set(term for terms in documents.values() for term in terms)

tfidf_term_embeddings = {term: get_embeddings(term, model, tokenizer) for term in unique_tfidf_terms}

vocab_keywords = set(keyword for keywords in vocabulary.values() for keyword in keywords)
vocabulary_embeddings = {keyword: get_embeddings(keyword, model, tokenizer) for keyword in vocab_keywords}

# Initialize a dictionary to hold the final tags for each document
final_tags = {}

# Loop over each document and its TF-IDF terms
for doc_id, terms in documents.items():
    doc_tags = {}
    for term in terms:
        term_embedding = tfidf_term_embeddings[term]
        
        similarities = {vocab_term: compute_similarity(term_embedding, vocabulary_embeddings[vocab_term]) for vocab_term in vocab_keywords}
        filtered_matches = {vocab_term: score for vocab_term, score in similarities.items() if score >= 0.80}
        
        # Select top N matches (if needed) after filtering
        top_matches = sorted(filtered_matches.items(), key=lambda item: item[1], reverse=True)[:3]  # Adjust N as needed
        
        # Store the matches for the current term
        doc_tags[term] = top_matches

    # Store the tags for the current document
    final_tags[doc_id] = doc_tags

# print(final_tags)

with open("tags.json", "w") as tags_file:
    json.dump(final_tags, tags_file, indent=4)
