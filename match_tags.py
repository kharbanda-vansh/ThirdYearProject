import json
from transformers import AutoModel, AutoTokenizer
import torch
from scipy.spatial.distance import cosine
import numpy as np

# Function to load model and tokenizer
def load_model_tokenizer(model_name="dmis-lab/biobert-v1.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

# Function to get embeddings
# Function to get embeddings with adjustment to ensure output is 1D
def get_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    # Use mean pooling and squeeze to get a 1D representation
    embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze().detach().numpy()
    return embeddings


# Function to compute cosine similarity
def compute_similarity(embedding1, embedding2):
    # Ensure embeddings are 1D
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()
    return 1 - cosine(embedding1, embedding2)

# Load model and tokenizer
model, tokenizer = load_model_tokenizer()

with open("vocabulary.json") as vocab_file:
    vocabulary = json.load(vocab_file)

with open("relevant_terms.json") as rel_terms_file:
    documents = json.load(rel_terms_file)
    

# Combine all unique TF-IDF terms from all documents into a set for efficiency
unique_tfidf_terms = set(term for terms in documents.values() for term in terms)

# Compute embeddings for all unique TF-IDF terms
tfidf_term_embeddings = {term: get_embeddings(term, model, tokenizer) for term in unique_tfidf_terms}

# Also, pre-compute embeddings for all unique vocabulary keywords (if not done already)
vocab_keywords = set(keyword for keywords in vocabulary.values() for keyword in keywords)
vocabulary_embeddings = {keyword: get_embeddings(keyword, model, tokenizer) for keyword in vocab_keywords}

# Initialize a dictionary to hold the final tags for each document
final_tags = {}

# Loop over each document and its TF-IDF terms
for doc_id, terms in documents.items():
    doc_tags = {}
    for term in terms:
        # Retrieve pre-computed embedding for the current term
        term_embedding = tfidf_term_embeddings[term]
        
        # Compute similarity with all vocabulary terms, filtering by threshold
        similarities = {vocab_term: compute_similarity(term_embedding, vocabulary_embeddings[vocab_term]) for vocab_term in vocab_keywords}
        filtered_matches = {vocab_term: score for vocab_term, score in similarities.items() if score >= 0.80}
        
        # Select top N matches (if needed) after filtering
        top_matches = sorted(filtered_matches.items(), key=lambda item: item[1], reverse=True)[:3]  # Adjust N as needed
        
        # Store the matches for the current term
        doc_tags[term] = top_matches

    # Store the tags for the current document
    final_tags[doc_id] = doc_tags

print(final_tags)



# # Sample terms (TF-IDF terms and vocabulary keywords)
# tfidf_terms = ['gene expression', 'DNA sequencing']
# vocabulary = {
#     "Transcriptomics": ["RNA", "transcripts", "expression"],
#     "Genomics": ["DNA", "genomes", "sequencing"]
# }

# # Calculate embeddings for TF-IDF terms
# tfidf_embeddings = {term: get_embeddings(term, model, tokenizer) for term in tfidf_terms}

# # Calculate embeddings for vocabulary keywords and flatten the structure for simplicity
# vocabulary_embeddings = {}
# for keyword, terms in vocabulary.items():
#     for term in terms:
#         vocabulary_embeddings[f"{keyword}: {term}"] = get_embeddings(term, model, tokenizer)

# # Compute similarities and select top matches
# top_matches = {}
# for tfidf_term, tfidf_embedding in tfidf_embeddings.items():
#     similarities = {keyword: compute_similarity(tfidf_embedding, vocab_embedding) for keyword, vocab_embedding in vocabulary_embeddings.items()}
#     filtered_matches = {keyword: score for keyword, score in similarities.items() if score >= 0.80}
#     # Sort by similarity score in descending order and select top N matches
#     top_matches[tfidf_term] = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:3]  # Adjust N as needed

# # Print top matches for each TF-IDF term
# for term, matches in top_matches.items():
#     print(f"Top matches for '{term}':")
#     for match in matches:
#         print(f"  {match[0]} with similarity {match[1]}")
