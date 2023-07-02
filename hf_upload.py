from huggingface_hub import HfApi
import os
import argparse

api = HfApi()

tokenizer_files = ["tokenizer.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.model"]
model_files = ["adapter_config.json", "adapter_model.bin"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_url", type=str, help="model url")
    parser.add_argument("--root_dir", type=str, help="model url")
    parser.add_argument("--checkpoint", type=str, help="checkpoint id")

    args = parser.parse_args().__dict__
    root = args.get("root_dir")

    files = [os.path.join(args.get("root_dir"), file) for file in tokenizer_files] + \
        [os.path.join(args.get("root_dir") ,args.get("checkpoint"), file) for file in model_files]
        
    for file in files:
        try:
            api.upload_file(
                path_or_fileobj=file,
                repo_id=args.get("model_url"),
                repo_type="model",
                path_in_repo=file.split('/')[-1]
            )
        except Exception as e:
            print(e)