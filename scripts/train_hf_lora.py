# adapted from https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/sample_finetune.py


import logging
import multiprocessing as mp
import os
import sys

from peft import LoraConfig

os.environ["TRITON_HOME"] = "./.cache/triton"
os.environ["TRITON_CACHE_DIR"] = "./.cache/triton_cache"
os.environ["HF_HOME"] = "./.cache/"
os.environ["HF_DATASETS_CACHE"] = "./.cache/"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_HOME"] = "./.cache/"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import datasets
import fire
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from src.constants import CHECKPOINT_PATH, HF_TOKEN

N_PROC = mp.cpu_count()
CONTEXT_LEN = 1024
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def apply_chat(elements, tokenizer):
    en = elements["en"]
    pt = elements["pt"]
    elements["text"] = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": f"Translate this text from English to European Portuguese: {en}"},
            {"role": "system", "content": pt},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    return elements


def _load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = CONTEXT_LEN
    # tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    return tokenizer


def load_data():
    dataset = datasets.load_dataset(
        "u1537782/PTradutor",
        "superclean",
    )
    return dataset["train"], dataset["valid"]


def tokenize_data(dataset, tokenizer):
    def tokenize(elements):
        chats = [
            [
                {"role": "user", "content": f"Translate this text from English to European Portuguese: {en}"},
                {"role": "system", "content": pt},
            ]
            for en, pt in zip(elements["en"], elements["pt"])
        ]
        inputs = tokenizer.apply_chat_template(
            chats,
            tokenize=True,
            return_tensors="pt",
            max_length=CONTEXT_LEN,
            truncation=True,
            padding=True,
            return_dict=True,
        )
        return inputs

    column_names = list(dataset.features)
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=column_names, num_proc=N_PROC)
    return tokenized_dataset


def main(
    model_name: str,
    output_folder: str,
    n_epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    early_stopping_patience: int,
    learning_rate: float = 2e-5,
    attn_implementation: str = "flash_attention_2",
    gradient_accumulation_steps=64,
):
    tokenizer = _load_tokenizer(model_name)

    trainset, validset = load_data()

    tokenized_trainset = tokenize_data(trainset, tokenizer)
    tokenized_validset = tokenize_data(validset, tokenizer)

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # BitsAndBytesConfig int-4 config
    # bnb_config = transformers.BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
        # quantization_config=bnb_config,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        # low_cpu_mem_usage=True
    )

    args = SFTConfig(
        output_dir=CHECKPOINT_PATH / output_folder,
        # train
        num_train_epochs=n_epochs,
        bf16=True,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=5_00,
        lr_scheduler_type="cosine",
        # eval
        do_eval=True,
        bf16_full_eval=True,
        eval_strategy="steps",
        eval_steps=500,
        per_device_eval_batch_size=eval_batch_size,
        # data
        dataloader_num_workers=N_PROC,
        dataset_text_field="text",
        # hf hub
        push_to_hub=True,
        hub_token=HF_TOKEN,
        hub_private_repo=True,
        # wandb
        run_name=output_folder,
        # dataset
        max_seq_length=CONTEXT_LEN,
        dataset_num_proc=N_PROC,
        # misc
        seed=42,
        data_seed=42,
        logging_steps=1,
        save_steps=500,
        save_total_limit=1,  # only stores the best model
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    peft_config = {
        "r": 128,
        "lora_alpha": 256,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear",
        "modules_to_save": None,
    }
    peft_conf = LoraConfig(**peft_config)
    log_level = args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}"
        + f" distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args}")
    trainer = SFTTrainer(
        model=model,
        args=args,
        peft_config=peft_conf,
        data_collator=data_collator,
        train_dataset=tokenized_trainset,
        eval_dataset=tokenized_validset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    fire.Fire(main)
