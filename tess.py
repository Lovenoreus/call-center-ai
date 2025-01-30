import yaml

with open("config2.yaml", "r") as file:
    try:
        config = yaml.safe_load(file)
        print("YAML parsed successfully:", config)
    except yaml.YAMLError as e:
        print("Error parsing YAML:", e)
