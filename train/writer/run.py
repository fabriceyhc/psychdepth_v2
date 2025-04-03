from ruamel.yaml import YAML
import os
import argparse
import traceback
import subprocess
from datetime import datetime

def update_config(base_config_path, updated_config_path, updates_dict):
    yaml = YAML()
    yaml.preserve_quotes = True

    with open(base_config_path, "r") as f:
        config = yaml.load(f)
    
    for key, value in updates_dict.items():
        if value is not None:
            config[key] = value

    # Save modified training config
    os.makedirs(os.path.dirname(updated_config_path), exist_ok=True)
    with open(updated_config_path, "w") as f:
        yaml.dump(config, f)

def main(args):

    # 1. Update training config
    if not args.updated_train_config_path:
        base_path = os.path.dirname(args.base_train_config_path)
        args.updated_train_config_path = os.path.join(
            base_path, 
            args.model_name, 
            f"train_lora_{args.stage}_{args.datasets}.yaml")

    if not args.train_output_dir:
        args.train_output_dir = os.path.join(
            args.cache_dir, 
            "lora",
            args.model_name, 
            args.datasets,
            args.stage)
            
    train_updates = {
        "model_name_or_path": args.model_name,
        "stage": args.stage,
        "dataset": args.datasets,
        "template": args.template,
        "num_train_epochs": args.num_train_epochs,
        "output_dir": args.train_output_dir
    }
    update_config(
        args.base_train_config_path, 
        args.updated_train_config_path, 
        train_updates
    )
  
    # 2. Run training command
    train_cmd = ["llamafactory-cli", "train", args.updated_train_config_path]
    print(f"Running training command: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True)

    # 3. Update export config
    if not args.updated_export_config_path:
        base_path = os.path.dirname(args.base_export_config_path)
        args.updated_export_config_path = os.path.join(
            base_path, 
            args.model_name, 
            f"export_lora_{args.stage}_{args.datasets}.yaml")

    if not args.export_output_dir:
        args.export_output_dir = os.path.join(
            args.cache_dir, 
            "merged",
            args.model_name, 
            args.datasets,
            args.stage)
            
    export_updates = {
        "model_name_or_path": args.model_name,
        "template": args.template,
        "adapter_name_or_path": args.train_output_dir,
        "export_dir": args.export_output_dir,
        "export_size": args.export_size
    }
    update_config(
        args.base_export_config_path, 
        args.updated_export_config_path, 
        export_updates
    )

    # 3. Run export command
    export_cmd = ["llamafactory-cli", "export", args.updated_export_config_path]
    print(f"Running export command: {' '.join(export_cmd)}")
    subprocess.run(export_cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaMA Factory Training Automation")
    
    # Config paths
    parser.add_argument("--base_train_config_path",  
                       default="./train/writer/configs/base_train_lora.yaml",
                       help="Path to base training config YAML")
    parser.add_argument("--updated_train_config_path", 
                       help="Path to save modified training config")
    parser.add_argument("--base_export_config_path", 
                       default="./train/writer/configs/base_export_lora.yaml",
                       help="Path to export config YAML")
    parser.add_argument("--updated_export_config_path", 
                       help="Path to save modified export config")

    # Model parameters
    parser.add_argument("--model_name", required=True,
                       help="Model identifier or path")
    parser.add_argument("--cache_dir", default='/data2/.shared_models',
                       help="Directory for storing base models")
    parser.add_argument("--stage", default="sft", 
                       help="Training stage (sft, dpo, kto)")
    parser.add_argument("--datasets", required=True,
                       help="Comma-separated list of datasets")
    parser.add_argument("--template", required=True,
                       help="Prompt template to use")
    parser.add_argument("--num_train_epochs", default=3,
                       help="Number of epochs to train")
    parser.add_argument("--train_output_dir", 
                       help="Output directory for trained lora model")
    parser.add_argument("--export_output_dir", 
                       help="Output directory for final merged model")
    parser.add_argument("--export_size", default=1,
                       help="Number of shards for the final model")

    args = parser.parse_args()
    
    try:
        main(args)
        print("Training and export completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {traceback.format_exc()}")
    except Exception as e:
        print(f"An error occurred: {str(traceback.format_exc())}")