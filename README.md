## This is the repository for third year project of Vansh
## Title of the project -  Improving search for the TESS Portal

## Work Plan:
The Complete project is divided into two main parts:

1. Improving tagging of all the entries int the portal
2. Improving Solr search engine

### Improving Tags:
1. Establish a base line of tags present in different sections as well overall tags present:
    - Get an average tags of all the tags present in the portal
    - Get an average tags of tags for each section
2. Get the data dump from bio.tools to form a vocabulary for tags
3. Run text scrappers to find the appropiate tags for all modules
4. Test the optimal number of tags
5. Devise a way to add tags whenever a new module is added in order to maintain the integrity of the search

### Adding the knowlwdge graphs:
1. Based on the tags find similarity between all events, materials and worflows
2. Set a threshold and find related events, materials and worflows
3. Add the knowledge graph in every page