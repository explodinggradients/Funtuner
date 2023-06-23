from funtuner.inference import Inference
from datasets import load_dataset
import argparse

def sampling(examples, model, generation_args):
    
    examples = [[example["instruction"],example["input"]] for example in examples]
    output = model.batch_generate(examples, **generation_args)
    return {"completion": output}
    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_url", type=str, default="shahules786/GPTNeo-125M-lora", help="model name")
    parser.add_argument("--dataset", type=str, default="vicgalle/alpaca-gpt4", help="dataset name")
    parser.add_argument("--num_samples", type=int, default=100, help="num of samples to run inference")
    parser.add_argument("--hf_upload", type=int, default="shahues786/test-results", help="hf repo")

    parser.add_argument("--batch_size", type=int, default=100, help="")
    parser.add_argument("--temperature", type=float, default=0.1, help="")
    parser.add_argument("--top_p", type=float, default=0.75, help="")
    parser.add_argument("--top_k", type=int, default=40, help="")
    parser.add_argument("--num_beams", type=int, default=4, help="")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="")

    generation_args = ["temperature", "top_p", "top_k", "num_beams", "max_new_tokens"]


    args = parser.parse_args().__dict__
    
<<<<<<< HEAD
    model = Inference(args.get("model"))
    dataset = load_dataset(args.get("dataset"), split="train").select(range(0, 100))
    dataset.map(sampling, batch_size=args.get("batch_size"))
    
=======
    generation_args = {k: args.get(k) for k in generation_args}
    model = Inference(args.get("model"))
    dataset = load_dataset(args.get("dataset"), split="train").select(range(0, 100))
    dataset = dataset.map(lambda batch: sampling(batch, model, generation_args), batch_size=args.get("batch_size"))
    dataset.push_to_hub(args.get("hf_upload"))
>>>>>>> dev

    