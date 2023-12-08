'''
This file gets the current tags for all the modules in the TESS portal
'''

import time
import urllib.request as request
import json


def get_tags(url, tags_dict):

    headers = {'Accept': 'application/vnd.api+json'}

    req = request.Request(url, headers=headers)

    with request.urlopen(req) as response:
        source = response.read()
        data = json.loads(source)
        if data:
            id_keywords_dict = {}
            for i in data['data']:
                id_keywords_dict[i['id']] = i['attributes']['keywords']
            tags_dict = tags_dict | id_keywords_dict # Merging the dictionary into main dictionary
        return data, tags_dict


def iterate_pages(mod):
    base_link = 'https://tess.elixir-europe.org'
    # curr = '/events?page_number=15'
    link = base_link + mod
    final_tags = {}
    # print(final_tags)
    data, final_tags = get_tags(link, final_tags)
    # print(final_tags)
    last = data['links']['last']

    while True:
        curr = data['links']['next']
        # print(curr)
        link = base_link + curr
        data, final_tags = get_tags(link, final_tags)
        if curr == last:
            break
    return final_tags


def write_data(final_tags, mod):
    with open(f'{mod}_data.json', 'w') as data_file:
        try:
            json.dump(final_tags, data_file, indent=2)
            print(f'data write successful - {mod}')
            print(len(final_tags))
        except Exception as e:
            print(f'something went wrong - data write - {e}')


# running the code for events
start = time.time()
tags = iterate_pages('/events')
write_data(tags, 'events')
print(f'Time taken- {time.time()-start}')

start = time.time()
tags = iterate_pages('/workflows')
write_data(tags, 'workflows')
print(f'Time taken- {time.time()-start}')

start = time.time()
tags = iterate_pages('/materials')
write_data(tags, 'materials')
print(f'Time taken- {time.time()-start}')


# base_link = 'https://tess.elixir-europe.org'
# curr = '/events?page_number=15'
# link = base_link + curr 
# link = 'https://tess.elixir-europe.org/events.json_api' + '?page_number=2'

# data = get_tags(link)

# with open('data.json', 'w') as data_file:
#         if data:
#             # id_keywords_dict = {}
#             # for i in data['data']:
#             #     id_keywords_dict[i['id']] = i['attributes']['keywords']
#             # json.dump(id_keywords_dict, data_file, indent=2)
#             json.dump(data['data'], data_file, indent=2)
#             print('data processed')
#         else:
#              print('something went wrong')
# print(type(data))
# print(data['links'])
# print(data['meta']['results-count'])
            # id_keywords_dict = {}
            # for i in data['data']:
            #     id_keywords_dict[i['id']] = i['attributes']['keywords']