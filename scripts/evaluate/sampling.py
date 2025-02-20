"""python scripts/evaluate_translation.py"""

import json
import logging
import os

import datasets
import fire
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.constants import CACHE_PATH, RESULTS_PATH
from src.metrics import BLEU, BLEURT, COMET, ROUGE, VIdScore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")

os.environ["HF_HOME"] = "./.cache/"
os.environ["HF_DATASETS_CACHE"] = "./.cache/"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_HOME"] = "./.cache/"


def _load_frmt(sample: int = None):
    frmt = datasets.load_dataset("u1537782/frmt", split="test")
    frmt = frmt.filter(lambda x: x["pt"] is not None)
    frmt = frmt.remove_columns(["br", "tw", "cn", "tw_simplified"])
    frmt = frmt.to_pandas().drop_duplicates()
    frmt = datasets.Dataset.from_pandas(frmt)
    if sample:
        frmt = frmt.select(range(sample))
    return frmt


def _load_data(dataset_name: str) -> datasets.Dataset:
    if dataset_name == "frmt":
        return _load_frmt()


def _evaluate(predictions: list[str], target: list[str], sources: list[str]) -> dict:
    return {
        "bleu": BLEU().compute(predictions=predictions, references=target),
        "rouge": ROUGE().compute(predictions=predictions, references=target),
        "comet": COMET().compute(predictions=predictions, references=target, sources=sources),
        "vid_score": VIdScore().compute(predictions=predictions, references=target),
        # "bleurt": BLEURT().compute(predictions=predictions, references=target),
    }


def _translate(
    texts: list[str],
    model_name: str,
    num_beams: int = None,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: float = 0,
    temperature: float = 1,
) -> list[str]:
    logging.info("Loading model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).eval()

    logging.info("Preparing data")
    if "llama" in model_name:
        formatted_prompts = [
            [
                [
                    {"role": "system", "content": "You are a translator from English to European Portuguese"},
                    {"role": "user", "content": f"Translate this text from English to European Portuguese: {text}"},
                ]
            ]
            for text in texts
        ]
    else:
        formatted_prompts = [
            [
                [
                    {"role": "user", "content": f"Translate this text from English to European Portuguese: {text}"},
                ]
            ]
            for text in texts
        ]

    tokenized_prompts = [
        tokenizer.apply_chat_template(
            formatted_prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True,
            return_dict=True,
        )
        for formatted_prompt in formatted_prompts
    ]

    logging.info("Translating dataset")
    sample = True if top_p != 1.0 or top_k != 50 else False
    with torch.no_grad():
        completions = []
        for batch in tqdm(tokenized_prompts):
            batch = batch.to(model.device)
            outputs = model.generate(
                **batch,
                max_length=1024,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                top_p=top_p,
                top_k=top_k,
                do_sample=sample,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                temperature=temperature,
            )
            input_len = batch["input_ids"].shape[1]
            generated_token_ids = outputs[:, input_len:]
            generated_text = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
            completions.extend(generated_text)
            print(generated_text)

    return completions


def main(
    dataset_name: str = "frmt",
    model_name: str = "u1537782/hf_phi3_fft",
    num_beams: int = 1,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: float = 0,
    temperature: float = 1,
):
    logging.info("Loading data.")
    dataset = _load_data(dataset_name)

    logging.info("Loading model.")
    translation = _translate(
        texts=dataset["en"],
        model_name=model_name,
        num_beams=num_beams,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        temperature=temperature,
    )

    model_stem = str(model_name).split("/")[-1]
    filename = f"{model_stem}_nb_{num_beams}_p_{top_p}_k_{top_k}_rep_p_{repetition_penalty}_len_p_{length_penalty}_temp_{temperature}_ngram-{no_repeat_ngram_size}.json"

    cache_fp = CACHE_PATH / dataset_name / filename
    if not cache_fp.parent.exists():
        cache_fp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(translation, cache_fp.open("w"), indent=4)

    logging.info("Evaluating translation")
    result = _evaluate(predictions=translation, target=dataset["pt"], sources=dataset["en"])

    logging.info("Storing results")
    results_fp = RESULTS_PATH / dataset_name / filename
    if not results_fp.parent.exists():
        results_fp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, results_fp.open("w"), indent=4)
    logging.info(f"Results stored in {results_fp}")


if __name__ == "__main__":
    fire.Fire(main)
