import json

with open('events_data.json') as events_file:
    events_data = json.load(events_file)
    tags = events_data.values()
    total_count = 0
    for i in tags:
        total_count += len(i)
    print("Events:")
    print(f"Total number of tags: {total_count} \nnumber of events present: {len(events_data)} \naverage number of tags: {total_count/len(events_data)}")

with open('workflows_data.json') as workflows_file:
    workflows_data = json.load(workflows_file)
    tags = workflows_data.values()
    total_count = 0
    for i in tags:
        total_count += len(i)
    print("Events:")
    print(f"Total number of tags: {total_count} \nnumber of events present: {len(workflows_data)} \naverage number of tags: {total_count/len(workflows_data)}")

with open('materials_data.json') as materials_file:
    materials_data = json.load(materials_file)
    tags = materials_data.values()
    total_count = 0
    for i in tags:
        total_count += len(i)
    print("Events:")
    print(f"Total number of tags: {total_count} \nnumber of events present: {len(materials_data)} \naverage number of tags: {total_count/len(materials_data)}")