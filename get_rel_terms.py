# from fuzzywuzzy import fuzz
import json
import urllib.request as request

import re
from urllib.error import HTTPError
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
import string
import contractions
# from nltk.stem import PorterStemmer

# from collections import defaultdict

import nltk
nltk.download('stopwords')


# website data
def fetch_website_data():
    dataset = []
    for x in range(1,270):
        url = f'https://tess.elixir-europe.org/events/{x}/'
        headers = {'Accept': 'application/vnd.api+json'}
        req = request.Request(url, headers=headers)
        try:
            with request.urlopen(req) as response:
                source = response.read()
                data = json.loads(source)
                website_data = data['data']['attributes']['title']
                if data['data']['attributes']['description'] is not None:
                    website_data += data['data']['attributes']['description']
                dataset.append(website_data)
        except HTTPError as e:
            if e.code == 404:
                print(x, "Event not found")
                dataset.append(" ")
            else:
                print("Some other error fetching event", x)
                dataset.append(" ")
    return dataset

    
def is_english(word):
    return bool(re.match(r'^[a-zA-Z]+$', word))

def clean_text(text):
    # converting all text to lowercase
    text = text.lower()
    # removing all the numbers
    text = re.sub(r'\d+', '', text)
    # removing the punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # removing any extra spaces
    text = re.sub(' +', ' ', text)
    # filtering out stopwords and non-english words
    word_tokens = word_tokenize(text)

    english_words = [word for word in word_tokens if is_english(word)]
    
    text = ' '.join([word for word in english_words if word not in stop_words])

    text = contractions.fix(text)

    return text


def retrieve_relevant_terms(clean_dataset):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(clean_dataset)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    term_dictionary = {}
    # Fetching relevant terms to add tags based on them
    for i, document in enumerate(clean_dataset):
        document_tfidf_scores = tfidf_matrix[i, :].toarray()[0]
        # Create a dictionary to store word and its corresponding TF-IDF score
        word_tfidf_dict = {feature_names[j]: document_tfidf_scores[j] for j in range(len(feature_names)) if document_tfidf_scores[j] > 0}
        # Sort the words based on their TF-IDF scores in descending order
        sorted_words = sorted(word_tfidf_dict.items(), key=lambda x: x[1], reverse=True)
        # Select words, up to 10, ensuring they have a positive TF-IDF score
        relevant_words = [word for word, score in sorted_words[:10]]
        
        if (len(relevant_words) == 0):
            term_dictionary[i+1] = []
        else:
            term_dictionary[i+1] = relevant_words

    return term_dictionary




stop_words = set(stopwords.words('english'))

dataset = fetch_website_data()
clean_dataset = [clean_text(i) for i in dataset]
term_dictionary = retrieve_relevant_terms(clean_dataset)
with open('relevant_terms.json', 'w') as vocab_file:
    json.dump(term_dictionary, vocab_file, indent=4)
print(f'Total number of events parsed: {len(term_dictionary)}')


























# def preprocess(text_list):
#     preprocessed_words = []
#     for text in text_list:
#         tokens = word_tokenize(text.lower())
#         tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
#         preprocessed_words.extend(tokens)
#     return ' '.join(preprocessed_words)



# with open("vocabulary.json", "r") as vocab_file:
#     vocabulary = json.load(vocab_file)

# stemmer = PorterStemmer()



# similar_words_dict = vocabulary
# documents_top_terms = term_dictionary

# similar_words_texts = [preprocess(similar_words) for similar_words in similar_words_dict.values()]

# # Vectorize the similar words dictionary
# vectorizer = TfidfVectorizer()
# similar_words_matrix = vectorizer.fit_transform(similar_words_texts)
# similar_words_index = vectorizer.get_feature_names_out()

# print("Size of similar_words_index:", len(similar_words_index))
# print("Number of features in TF-IDF matrix:", similar_words_matrix.shape[1])  # Get the number of features


# # # Calculate similarity scores
# # similar_words_scores = defaultdict(dict)
# # for doc, terms in documents_top_terms.items():
# #     for term in terms:
# #         if term in similar_words_dict:
# #             term_vector = vectorizer.transform([preprocess(similar_words_dict[term])])
# #             similarity_scores = cosine_similarity(term_vector, similar_words_matrix)
# #             similar_words_scores[doc][term] = {
# #                 similar_words_index[idx]: score 
# #                 for idx, score in enumerate(similarity_scores.flatten())
# #             }

# # Calculate similarity scores
# similar_words_scores = defaultdict(dict)
# for doc, terms in documents_top_terms.items():
#     for term in terms:
#         if term in similar_words_dict:
#             term_vector = vectorizer.transform([preprocess(similar_words_dict[term])])
#             similarity_scores = cosine_similarity(term_vector, similar_words_matrix)
#             similar_words_scores[doc][term] = {
#                 similar_words_index[min(idx, len(similar_words_index) - 1)]: score  # Limiting index to the length of similar_words_index
#                 for idx, score in enumerate(similarity_scores.flatten())
#             }

# # Select top similar words
# top_similar_words = defaultdict(dict)
# for doc, terms_scores in similar_words_scores.items():
#     for term, scores in terms_scores.items():
#         sorted_similar_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]  # Adjust the number of top similar words
#         top_similar_words[doc][term] = sorted_similar_words

# # Print the top similar words for each document and term
# for doc, terms in top_similar_words.items():
#     print(f"Document: {doc}")
#     for term, similar_words in terms.items():
#         print(f"Top similar words for '{term}': {similar_words}")
