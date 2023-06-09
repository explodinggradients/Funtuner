import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import Trainer
from funtuner.custom_datasets import get_datasets, FunDataCollator
from funtuner.utils import get_model, get_name, get_tokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training


class FunTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        
        outputs = model(**inputs)
        loss = outputs.get("loss")
        return (loss, outputs) if return_outputs else loss


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    if not os.path.exists(cfg.log_dir):
        os.mkdir(cfg.log_dir)

    if not cfg.log_wandb:
        os.environ["WANDB_MODE"] = "offline"

    if cfg.log_wandb:
        import wandb
        if cfg.run_name == "":
            cfg.run_name = get_name()
        name = f"{cfg.model.split('/')[-1]}-{cfg.run_name}"
        wandb.init(
            project="Funtuner",
            entity=cfg.wandb_entity,
            name=name,
            config=cfg,
        )

    model = get_model(cfg.model)
    tokenizer = get_tokenizer(cfg)
    model.resize_token_embeddings(len(tokenizer))
    
    if cfg.eight_bit_training:
        model = prepare_model_for_int8_training(model)

    if cfg.LoRa:
        Lora_config = LoraConfig(
            r=cfg.LoraConfig.get("r", 8),
            target_modules=cfg.LoraConfig("target_modules", ["q_proj", "v_proj"]),
            lora_alpha=cfg.LoraConfig("lora_alpha", 16),
            lora_dropout=cfg.LoraConfig("lora_dropout", 0.05),
            fan_in_fan_out=cfg.LoraConfig(
                "fan_in_fan_out",
            ),
            bias=cfg.LoraConfig("bias", "none"),
        )

        model = get_peft_model(model, Lora_config)
        print("--------LoRA------------")
        model.print_trainable_parameters()


    training_args = instantiate(
        cfg.trainer,
        deepspeed=cfg.deepspeed_config if cfg.deepspeed else None,
        report_to="wandb" if cfg.log_wandb else None,
    )
    train_dataset, validation_dataset = get_datasets(cfg)

    datacollator = FunDataCollator(
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )

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

    trainer.save_model(os.path.join(cfg.log_dir, f"{cfg.model.split('/')[-1]}-model"))
    tokenizer.save_pretrained(cfg.log_dir)


if __name__ == "__main__":
    train()
