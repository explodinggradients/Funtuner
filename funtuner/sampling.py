import itertools
from funtuner.inference import Inference
from funtuner.custom_datasets.utils import DATASET_MAPPING
from datasets import load_dataset
import argparse

def sampling(examples, model, dataset, **generation_args):
    
    dataset = DATASET_MAPPING[dataset]
    instruction, input = dataset.get("prompt"), dataset.get("context",None)
    instruction = examples[instruction]
    input = examples[input] if input is not None else [None]
    examples =  list(itertools.zip_longest(instruction, input))
    output = model.batch_generate(examples, **generation_args)
    return {"completion": output}
    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_url", type=str, help="model name")
    parser.add_argument("--load_8_bits", type=bool, default=False, help="model name")

    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--split", type=str, default="test", help="dataset split")

    parser.add_argument("--num_samples", type=int, default=100, help="num of samples to run inference")
    parser.add_argument("--save_path", type=str, default="results.json", help="save path")

    parser.add_argument("--batch_size", type=int, default=4, help="")
    parser.add_argument("--temperature", type=float, default=0.1, help="")
    parser.add_argument("--top_p", type=float, default=0.75, help="")
    parser.add_argument("--top_k", type=int, default=40, help="")
    parser.add_argument("--num_beams", type=int, default=4, help="")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="")

    generation_args = ["temperature", "top_p", "top_k", "num_beams", "max_new_tokens"]


    args = parser.parse_args().__dict__
    
    generation_args = {k: args.get(k) for k in generation_args}
    model = Inference(args.get("model_url"), load_in_8bit=args.get("load_8_bits"))
    dataset = load_dataset(args.get("dataset"), split=args.get("split")).select(range(0, args.get("num_samples")))
    dataset = dataset.map(lambda batch: sampling(batch, model, args.get("dataset"), **generation_args), batch_size=args.get("batch_size"), batched=True)
    dataset.to_json(args.get("save_path"), indent=4)

    