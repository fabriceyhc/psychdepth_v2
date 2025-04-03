from ruamel.yaml import YAML

def update_config(base_config_path, updated_config_path, updates_dict):
    with open(base_config_path, "r") as f:
        config = yaml.load(f)
    
    for key, value in updates_dict.items():
        if value is not None:
            config[key] = value

    # Save modified training config
    with open(updated_config_path, "w") as f:
        yaml.dump(config, f)