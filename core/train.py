import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
from peft import PromptTuningConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from I2CL.my_datasets import get_dataset
from src import (
    CustomWandbCallback,
    setup_logger,
    load_yml,
    CustomTrainer,
    CustomTrainingArguments,
    DataCollatorForSPT
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training with HuggingFace Trainer")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument('--wandb-key', type=str, required=False, help="weights & biases key.")

    args = parser.parse_args()

    return args


def main() -> None:
    # Parse arguments
    args = parse_args()

    # Setup logger
    setup_logger()

    training_config = load_yml(args.cfg_path)

    # Set seeds for reproducibility
    if training_config["seed"] is not None:
        seed_num = training_config["seed"]
        random.seed(seed_num)
        np.random.seed(seed_num)
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        # Make cudnn deterministic for reproducibility
        cudnn.deterministic = True
        cudnn.benchmark = False
        logger.info(f"Set random seed to {seed_num}")

    # Setup file logging
    file_name = args.cfg_path.split('/')[-1].replace('.yml', '').replace('.yaml', '')
    log_dir = os.getenv("LOG_DIR", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    from src.utils import now
    job_id = now()
    log_file = f'{log_dir}/{file_name}_{job_id}.log'

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Starting training with config: {args.cfg_path}")

    # Initialize wandb if key is provided
    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    tokenizer = AutoTokenizer.from_pretrained(**training_config["tokenizer"])
    tokenizer.padding_side = "left"

    train_dataset = get_dataset(**training_config["train_dataset_config"])
    eval_dataset = get_dataset(**training_config["eval_dataset_config"])

    data_collator = DataCollatorForSPT(
        tokenizer=tokenizer,
        dataset_class=train_dataset,
        **training_config["data_collator"]
    )
    ## END

    training_args = CustomTrainingArguments(**training_config["trainer"])

    trainer = CustomTrainer(
        model=AutoModelForCausalLM.from_pretrained(**training_config["model"]),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset[:500],
        data_collator=data_collator,
        peft_config=PromptTuningConfig(
            prompt_tuning_init_text=train_dataset.get_task_instruction(),
            **training_config["peft_config"],
        ),
    )

    # Add custom callbacks
    if args.wandb_key:
        trainer.add_callback(CustomWandbCallback())

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Save model
    logger.info("Training completed. Saving model...")
    trainer.save_model()

    if args.wandb_key:
        wandb.finish()

    logger.info("Training finished successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Cleanup
        if wandb.run is not None:
            wandb.finish()
        raise
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        # Cleanup
        if wandb.run is not None:
            wandb.finish()
        raise
