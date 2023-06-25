# FunTuner
A no nonsense easy to configure model fine-tuning framework for GPT based models that can get the job done in a memory and time efficient manner. 

:radioactive: Work in progress

## Components
 ✅hydra configuration

 ✅Deepspeed support

 ✅8 bit training 

 ✅LoRA using peft

 ✅Sequence bucketing

 ✅Inference

    ✅single
    ✅batch
    ❎stream

✅Supported Models

    ✅GPTNeoX - Redajajama, Pythia, etc
    ❎LLama
    ❎Falcon 

❎Flash attention 


## Train

* Using deepspeed

```bash
deepspeed funtuner/trainer.py
```

## Inference
```python
from funtuner.inference import Inference
model = Inference("shahules786/GPTNeo-125M-lora")
kwargs = {"temperature":0.1,
        "top_p":0.75,
        "top_k":5,
        "num_beams":2,
        "max_new_tokens":128,}

##single
output =model.generate("Which is a species of fish? Tope or Rope",**kwargs)

##batch
inputs = [["There was a tiger in the hidden"],["Which is a species of fish? Tope or Rope"]]
output = model.batch_generate(inputs,**kwargs)

```


## Sampling

```bash
python funtuner/sampling.py --model_url shahules786/Redpajama-3B-CoT --dataset Dahoas/cot_gsm8k 
```