import os
import json
import traceback
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def process_folder(folder_path, error_log):
    """
    Input: path of the folder with json files present

    Output: dictionary with "biotoolsID" as key and "collectionID" as value
    """
    vocabulary_data = {}

    for filename in os.listdir(folder_path,):
            # print(filename)
            if filename.endswith('biotools.json'):
                file_path = os.path.join(folder_path, filename)
                # print(file_path)

                # Read the contents of the biotools.json file
                with open(file_path, 'r') as file:
                    biotools_data = json.load(file)

                    # Extract biotoolsID and CollectionID(or topic)
                    try:
                        biotools_id = biotools_data.get('biotoolsID')
                        biotools_id = lemmatizer.lemmatize(biotools_id)
                        collection_id = biotools_data.get('CollectionID')
                        collection_id = lemmatizer.lemmatize(collection_id)
                        if collection_id is None:
                            collection_id = biotools_data.get('collectionID')
                            collection_id = lemmatizer.lemmatize(collection_id)
                        # if biotools_id == 'rabbit_in_a_hat':
                        #     print(f'-------{collection_id}------')
                        if collection_id is None:
                            collection_id = biotools_data.get('topic')
                            collection_id = lemmatizer.lemmatize(collection_id)
                            if collection_id:
                                terms = []
                                for i in range (len(collection_id)):
                                    terms.append(collection_id[i]['term'])
                            collection_id = terms                   
                        # biotools_id = biotools_data['biotoolsID']
                        # collection_id = biotools_data['CollectionID']
                        # print(biotools_id, collection_id)

                        # # Add to vocabulary_data dictionary
                        if (biotools_id is not None) and (collection_id is not None):
                            vocabulary_data[biotools_id] = collection_id
                    except Exception as e:
                        # print(f'Error: {e} in file: {file_path}')
                        error_log.write(f'Error: {e} in file: {file_path}\n')
                        traceback.print_exc(file=error_log)
                         
    return vocabulary_data

def iterate_folder():
    root_folder = f'../research-software-ecosystem-content/data'
    final_vocabulary = {}

    with open('error.log', 'w') as error_log:
        count = 0
        # Iterate over subdirectories in the root folder
        for subdir in os.listdir(root_folder):
            subdir_path = os.path.join(root_folder, subdir)
            # print(subdir_path)
            count += 1
            # if count > 100:
            #      break

            # Check if it's a directory
            if os.path.isdir(subdir_path):
                # Process the folder and update the final vocabulary data
                folder_data = process_folder(subdir_path, error_log)
                final_vocabulary.update(folder_data)
                # print(final_vocabulary)

    # # Write the final vocabulary data to 'vocabulary.json'
    with open('vocabulary.json', 'w') as vocab_file:

        json.dump(final_vocabulary, vocab_file, indent=4)

    print(f'Total number of folders parsed: {count}')

iterate_folder()
