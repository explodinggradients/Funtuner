import argparse
from datasets import load_dataset
import yaml
from pathlib import Path
from datetime import datetime
from funtuner.inference import Inference
from funtuner.utils import save_json
from funtuner.inference import Inference

DATA = "shahules786/llm-eval"

def merge_dicts(generation_args, default_args):
    
    for _, args_dict in generation_args.items():
        args = {k: v for k, v in default_args.items() if k not in args_dict.keys()}
        args_dict.update(args)
    return generation_args


def sampling(examples, model, generation_args):
    datadict = {}
    instruction = examples["instruction"]
    inputs = examples["input"]
    examples = list(zip(instruction, inputs))
    for key, args in generation_args.items():
        output = model.batch_generate(examples, **args)
        datadict[f"{key}_completion"] = output

    return datadict

def update_results(dataset, generation_args: dict, results: dict):
    
    for item in dataset:
        sample = []
        for _, key in enumerate(generation_args.keys()):
            sample.append({"sampling_config": key, 
                          "sampling_params": generation_args[key],
                          "outputs": [item[f"{key}_completion"]]})
        sample_dict = {"prompt": item["instruction"], "results": sample}
        results["prompts"].append(sample_dict)
    
    return results


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_url", type=str, help="model name")
    parser.add_argument("--batch_size", type=int, default=2, help="model name")
    parser.add_argument("--load_8_bits", type=bool, default=False, help="model name")
    args = parser.parse_args().__dict__

    model_name = args.get("model_url")
    load_8_bits = args.get("load_8_bits")
    model = Inference(model_name, load_8_bits)
    dataset = load_dataset(DATA, split="train").shuffle(seed=42)
    generation_args = yaml.safe_load(Path("evals/config/generation.yaml").read_text())
    default_args = generation_args.pop("defaults")
    generation_args = merge_dicts(generation_args, default_args)
    dataset = dataset.map(lambda batch: sampling(batch, model, generation_args),
                          batch_size=args.get("batch_size"), batched=True)

    results = {
        "model_name": model_name,
        "date": datetime.utcnow().isoformat(),
        "args":{
            "device":"cuda",
            "batch_size":args.get("batch_size"),
            "dataset":DATA
        },
        "prompts":[]
    }
    results = update_results(dataset, generation_args, results)
    model_name = model_name.split("/")[-1]
    save_json(f"results-{model_name}.json", results)