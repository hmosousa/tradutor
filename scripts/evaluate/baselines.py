import json
import logging

import fire

from src.constants import RESULTS_PATH
from src.metrics import BLEU, BLEURT, COMET, ROUGE, VIdScore, BERTSCORE
from src.utils import load_data, load_model


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


def _evaluate(predictions: list[str], target: list[str], sources: list[str]) -> dict:
    return {
        "bleu": BLEU().compute(predictions=predictions, references=target),
        "rouge": ROUGE().compute(predictions=predictions, references=target),
        "comet": COMET().compute(predictions=predictions, references=target, sources=sources),
        "vid_score": VIdScore().compute(predictions=predictions, references=target),
        # "bleurt": BLEURT().compute(predictions=predictions, references=target),
        # "bertscore": BERTSCORE().compute(predictions=predictions, references=target),
    }


def main(dataset: str = "frmt", model: str = "deepl", checkpoint: str = None):
    logging.info(f"Evaluating {model} on {dataset}")
    dataset = load_data(dataset)
    logging.info(f"Loaded {dataset.name} with size {len(dataset)}")

    logging.info(f"Loading {model}")
    model = load_model(model, checkpoint)
    logging.info(f"Loaded {model.name}")

    logging.info("Translating dataset")
    translation = model(dataset)

    filename = f"{checkpoint}.json" if checkpoint else f"{model.name}.json"
    results_fp = RESULTS_PATH / dataset.name / filename
    del model

    logging.info("Evaluating translation")
    result = _evaluate(predictions=translation, target=dataset.target, sources=dataset.source)

    logging.info("Storing results")
    if not results_fp.parent.exists():
        results_fp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, results_fp.open("w"), indent=4)
    logging.info(f"Results stored in {results_fp}")


if __name__ == "__main__":
    fire.Fire(main)
