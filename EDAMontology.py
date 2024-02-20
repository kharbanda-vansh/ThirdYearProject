from rdflib import Graph

# Path to your OWL file
owl_file_path = "EDAM_1.25.owl"

# Create a Graph object
g = Graph()

# Load the OWL file into the Graph
g.parse(owl_file_path, format="xml")

# Iterate over each triple in the graph and print it
for subject, predicate, obj in g:
    print(subject, predicate, obj)
