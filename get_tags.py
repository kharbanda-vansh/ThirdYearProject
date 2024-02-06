from fuzzywuzzy import fuzz
import json
import urllib.request as request

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
import string
import contractions

import nltk
nltk.download('stopwords')

# website data
def fetch_website_data(n):
    dataset = []
    for x in range (1, n+1):
        url = f'https://tess.elixir-europe.org/events/{x}/'
        headers = {'Accept': 'application/vnd.api+json'}
        req = request.Request(url, headers=headers)
        with request.urlopen(req) as response:
            source = response.read()
            data = json.loads(source)
            # print(data['data']['attributes']['title'])
            # print(data['data']['attributes']['description'])
            website_data = data['data']['attributes']['title']
            if data['data']['attributes']['description'] is not None:
                website_data += data['data']['attributes']['description']
            dataset.append(website_data)
    return dataset
            # print(website_data)
            # website_data = website_data.split()
            # print(website_data)

    

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
    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(text)

    english_words = [word for word in word_tokens if is_english(word)]
    
    text = ' '.join([word for word in english_words if word not in stop_words])

    text = contractions.fix(text)

    return text


def retrieve_relevant_terms(clean_dataset):

    # for i in range (0, len(clean_dataset)):
    #     print(i, clean_dataset[i])

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(clean_dataset)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    term_dictionary = {}
    # fetchinng relevant terms to add tags based on them
    for i, document in enumerate(clean_dataset):
        # feature_index = tfidf_matrix[i, :].nonzero()[1]
        # tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
        # for word_index, score in tfidf_scores:
        #     print(f"Word: {feature_names[word_index]}, TF-IDF Score: {score}")
        document_tfidf_scores = tfidf_matrix[i, :].toarray()[0]
        # Create a dictionary to store word and its corresponding TF-IDF score
        word_tfidf_dict = {feature_names[j]: document_tfidf_scores[j] for j in range(len(feature_names))}
        # Sort the words based on their TF-IDF scores in descending order
        sorted_words = sorted(word_tfidf_dict.items(), key=lambda x: x[1], reverse=True)
        # Select top 10 words
        top_10_words = [word for word, score in sorted_words[:10]]
        
        # Print the document and its top 10 words
        # print(f"Document {i+1}: {document}")
        # print("Top 10 words:")
        # print(top_10_words)
        # for word in top_10_words:
        #     print(f"- {word}: {word_tfidf_dict[word]}")
        # print()
        
        if len(document) != 0:
            term_dictionary[i+1] = top_10_words
        else:
            term_dictionary[i+1] = []
        
    return term_dictionary

dataset = fetch_website_data(10)
clean_dataset = [clean_text(i) for i in dataset]
term_dictionary = retrieve_relevant_terms(clean_dataset)
print(term_dictionary)
    
    




































# # vocabulary with related terms
# with open('vocabulary.json', 'r') as file:
#             vocabulary = json.load(file)
# # print(len(vocabulary))

# # Similarity threshold
# threshold = 100

# # Calculate similarities and assign tags
# tagged_data = []
# # vocab = []
# # for term, related_terms in vocabulary.items():
# #      vocab.append(term)
# vocab = vocabulary.keys()
     
# # for item in website_data:
# #        print(item)
# tags = []
# tokens = website_data.split()  # Tokenize website data

# # print(tokens)
# for t in vocab:
#     for word in tokens:
#         similarity_score = fuzz.partial_ratio(t, word)  # Use fuzzy matching
#         print(similarity_score)
#         if similarity_score >= threshold:
#             tags.append(t)
#             # print(t)
# # print(tags)
# with open('test.txt', 'w') as test_file:
#         for t in tags:
#             test_file.write(t)
