from funtuner.inference import Inference
from datasets import load_dataset


def sampling(examples):
    
    examples = [[example["instruction"],example["input"]] for example in examples]
    output = model.batch_generate(examples)
    return {"completion": output}
    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_url", type=str, default="", help="model name")
    parser.add_argument("--dataset", type=str, default="vicgalle/alpaca-gpt4", help="dataset name")
    parser.add_argument("--num_samples", type=int, default=100, help="num of samples to run inference")


    args = parser.parse_args().__dict__
    
    model = Inference(args.get("model"))
    dataset = load_dataset(args.get("dataset"), split="train").select(range(0, 100))
    dataset.map(sampling, batch_size=args.get("batch_size"))
    

    