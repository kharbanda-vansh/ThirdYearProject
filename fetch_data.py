import ruamel.yaml

openapi_spec = "https://tess.elixir-europe.org/"
# Parse the YAML content
# spec_dict = yaml.safe_load(openapi_spec)
spec_dict = yaml.load(openapi_spec, Loader=yaml.FullLoader)

# Access the tags for the /events endpoint
tags_for_events = spec_dict.get("paths", {}).get("/events", {}).get("get", {}).get("tags", [])

# Print the tags
print("Tags for /events endpoint:", tags_for_events)