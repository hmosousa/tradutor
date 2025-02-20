"""python scripts/evaluate_translation.py"""

import json
import logging

import datasets
from transformers.pipelines.pt_utils import KeyDataset
import fire
import torch
from tqdm import tqdm
from transformers import pipeline

from src.constants import CACHE_PATH, RESULTS_PATH
from src.metrics import BLEU, BLEURT, COMET, ROUGE, VIdScore

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


def _load_frmt(sample: int = None):
    frmt = datasets.load_dataset("u1537782/frmt", split="test")
    frmt = frmt.filter(lambda x: x["pt"] is not None)
    frmt = frmt.remove_columns(["br", "tw", "cn", "tw_simplified"])
    frmt = frmt.to_pandas().drop_duplicates()
    frmt = datasets.Dataset.from_pandas(frmt)
    if sample:
        frmt = frmt.select(range(sample))
    return frmt


def _load_ntrex(sample: int = None):
    pt = datasets.load_dataset("davidstap/NTREX", "por_Latn", split="test")["text"]
    en = datasets.load_dataset("davidstap/NTREX", "eng_Latn", split="test")["text"]
    ids = list(range(len(pt)))
    ntrex = datasets.Dataset.from_dict({"idx": ids, "en": en, "pt": pt})
    if sample:
        ntrex = ntrex.select(range(sample))
    return ntrex


def _load_data(dataset_name: str) -> datasets.Dataset:
    if dataset_name == "frmt":
        return _load_frmt()
    elif dataset_name == "ntrex":
        return _load_ntrex()
    else:
        raise ValueError(f"Dataset name {dataset_name} is not valid.")


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
) -> list[str]:
    logging.info("Loading model and tokenizer")

    pipe = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",  # replace with "mps" to run on a Mac device
    )

    logging.info("Preparing data")
    if "llama" in model_name:
        formatted_prompts = [
            [
                {"role": "system", "content": "You are a translator from English to European Portuguese"},
                {"role": "user", "content": f"Translate this text from English to European Portuguese: {text}"},
            ]
            for text in texts
        ]
    else:
        formatted_prompts = [
            [
                {"role": "user", "content": f"Translate this text from English to European Portuguese: {text}"},
            ]
            for text in texts
        ]

    logging.info("Translating dataset")
    translations = []
    for prompt in tqdm(formatted_prompts):
        outputs = pipe(prompt, max_new_tokens=1024)
        translation = outputs[0]["generated_text"][-1]["content"].strip()
        translations.append(translation)
        logging.debug(f"Translation: {translation}")

    return translations


def main(
    dataset_name: str = "frmt",
    model_name: str = "u1537782/hf_gemma_fft",
):
    logging.info("Loading data.")
    dataset = _load_data(dataset_name)

    logging.info("Loading model.")
    model_stem = str(model_name).split("/")[-1]
    filename = f"{model_stem}.json"
    cache_fp = CACHE_PATH / dataset_name / filename
    if not cache_fp.exists():
        cache_fp.parent.mkdir(parents=True, exist_ok=True)
        translation = _translate(
            texts=dataset["en"],
            model_name=model_name,
        )
        json.dump(translation, cache_fp.open("w"), indent=4)
    else:
        translation = json.load(cache_fp.open())

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
