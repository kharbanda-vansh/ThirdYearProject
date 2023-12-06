'''
This file gets the current tags for all the modules in the TESS portal
'''

import urllib.request as request
import json

def get_tags(url):
    with request.urlopen(url) as response:
        source = response.read()
        data = json.loads(source)
        return data

base_link = 'https://tess.elixir-europe.org'
curr = '/events'
link = base_link + curr + '.json_api'
data = get_tags(link)

with open('data.json', 'w') as data_file:
        if data:
            id_keywords_dict = {}
            for i in data['data']:
                id_keywords_dict[i['id']] = i['attributes']['keywords']
            json.dump(id_keywords_dict, data_file, indent=2)
            # json.dump(data, data_file, indent=2)
            print('data processed')
        else:
             print('something went wrong')
# print(type(data))
# print(data.keys())
# print(data['links'])
# print(data['meta'])