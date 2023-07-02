import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import Trainer, BitsAndBytesConfig
from funtuner.custom_datasets import get_datasets, FunDataCollator
from funtuner.utils import get_model, get_name, get_tokenizer, add_additional_config, get_lora_modules, prepare_model_types
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from omegaconf import OmegaConf
import torch
import warnings
warnings.filterwarnings("ignore")

JOB_ID = os.environ.get("SLURM_JOB_ID",None)
class FunTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.get("loss")
        return (loss, outputs) if return_outputs else loss


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    random_runname = get_name()
    if not os.path.exists(cfg.log_dir):
        os.mkdir(cfg.log_dir)
    if JOB_ID is not None:
        cfg.log_dir = os.path.join(cfg.log_dir, JOB_ID)
        if not os.path.exists(cfg.log_dir):
            os.mkdir(cfg.log_dir)
        

    if not cfg.log_wandb:
        os.environ["WANDB_MODE"] = "offline"

    if cfg.log_wandb:
        import wandb
        if cfg.run_name == "":
            cfg.run_name = random_runname
        name = f"{cfg.model.split('/')[-1]}-{cfg.run_name}"
        wandb.init(
            project="Funtuner",
            entity=cfg.wandb_entity,
            name=name,
            config=cfg,
        )
    print("DEVICES", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    model_args = {"load_in_4bit": cfg.load_in_4_bit, "load_in_8bit": cfg.load_in_8_bit}
    if cfg.qlora:
        compute_dtype = (torch.float16 if cfg.trainer.fp16 else (torch.bfloat16 if cfg.trainer.bf16 else torch.float32))
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=cfg.load_in_8_bit,
            load_in_4bit=cfg.load_in_4_bit,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg.qlora_config.double_quant,
            bnb_4bit_quant_type=cfg.qlora_config.quant_type,
        )
        model_args.update({"quantization_config":quantization_config, 
                           "torch_dtype":compute_dtype})
    
    model = get_model(cfg.model, **model_args)
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    tokenizer = get_tokenizer(cfg)
    model.resize_token_embeddings(len(tokenizer))
    
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
    if cfg.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.trainer.gradient_checkpointing)
        

    if cfg.LoRa:
        Lora_config = OmegaConf.to_object(cfg.LoraConfig)
        Lora_config["target_modules"] = get_lora_modules(model, cfg)
        Lora_config = LoraConfig(
            **Lora_config
        )

        model = get_peft_model(model, Lora_config)
        print("--------LoRA------------")
        model.print_trainable_parameters()
        
    model = prepare_model_types(model, cfg)


    training_args = instantiate(
        cfg.trainer,
        deepspeed=cfg.deepspeed_config if cfg.deepspeed else None,
        report_to="wandb" if cfg.log_wandb else None,
        output_dir=cfg.log_dir,

    )
    train_dataset, validation_dataset = get_datasets(cfg)

    datacollator = FunDataCollator(
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )
    
    # from tqdm import tqdm
    # for i,item in tqdm(enumerate(train_dataset)):
    #     output = datacollator([item])
    #     if not (output["input_ids"].size() == output["labels"].size() == output["attention_mask"].size()) :
    #         print("ERROR",i)
    #     if output['input_ids'].size(-1) > 512:
    #         print(output['input_ids'].size(-1))
    # Initialize our Trainer
    trainer = FunTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=datacollator,
    )

    # training
    trainer.train()

    # save 
    trainer.save_model(os.path.join(cfg.log_dir, f"{cfg.model.split('/')[-1]}-model"))    
    tokenizer.save_pretrained(cfg.log_dir)
    add_additional_config(cfg.log_dir)
    
if __name__ == "__main__":
    train()
