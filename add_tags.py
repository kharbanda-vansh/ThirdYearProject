# from requests import request
# import json
# import requests
import spacy
# from bs4 import BeautifulSoup
import time
import urllib.request as request
import json

# Load spaCy model (make sure to install it first: pip install spacy)
nlp = spacy.load("en_core_web_md")

def extract_tags_from_text(text, vocabulary):
    print('here')
    tags = set()
    # for main_word, related_words in vocabulary.items():
    #     print (main_word, related_words)
    #     # Check if main word or any related word is present in the text
    #     if any(word in text for word in [main_word] + related_words):
    #         print(main_word, related_words)
    #         tags.add(main_word)
    
        # Find similar words based on spaCy word similarity
        # for similar_word in find_similar_words(main_word, vocabulary):
        #     if similar_word in text:
        #         tags.add(similar_word)
    
    # return list(tags)
    main_word = vocabulary.keys()
    related_words =vocabulary.values()
    print(len(main_word) + len(related_words))
    for t in text:
        if t in main_word:
            tags.append()
    
def find_similar_words(word, vocabulary):
    similar_words = set()
    word_vector = nlp(word).vector

    for vocab_word, _ in vocabulary.items():
        vocab_vector = nlp(vocab_word).vector
        similarity = word_vector.dot(vocab_vector) / (nlp(word).similarity(nlp(vocab_word)) + 1e-8)

        # You can adjust the threshold based on your requirements
        if similarity > 0.7:
            similar_words.add(vocab_word)
    
    return list(similar_words)

def main():
    url = 'https://tess.elixir-europe.org/events/1/'
    # response = requests.get(url)

    # if response.status_code == 200:
    #     soup = BeautifulSoup(response.text, 'html.parser')
    #     # Extract text content from the HTML
    #     text_content = soup.get_text()
    #     print(text_content)
    headers = {'Accept': 'application/vnd.api+json'}

    req = request.Request(url, headers=headers)

    with request.urlopen(req) as response:
        source = response.read()
        data = json.loads(source)
        # print(data['data']['attributes']['title'])
        # print(data['data']['attributes']['description'])
        text_content = data['data']['attributes']['title'] + data['data']['attributes']['description']

        # Load vocabulary from the JSON file
        with open('vocabulary.json', 'r') as file:
            vocabulary = json.load(file)
            
        # # Extract tags based on the vocabulary and similar words
        # tags = extract_tags_from_text(text_content, vocabulary)
        # print(tags)
    # else:
    #     print(f'Error: {response.status_code}')

if __name__ == "__main__":
    main()
