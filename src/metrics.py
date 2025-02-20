from typing import Any

import evaluate
import numpy as np
import scipy.stats as stats
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def compute_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2.0, len(data) - 1)
    confidence_interval = [mean - margin_of_error, mean + margin_of_error]
    return mean, margin_of_error, confidence_interval


class Metric:
    def __call__(self, predictions: list[str], references: list[str], *args, **kwargs) -> Any:
        return self.compute(predictions, references, *args, **kwargs)

    def compute(self, predictions: list[str], references: list[str], *args, **kwargs) -> Any:
        raise NotImplementedError


class BLEU(Metric):
    def __init__(self) -> None:
        self._engine = evaluate.load("sacrebleu", smooth_method="exp", use_effective_order=False, tokenize="13a")

    def compute(self, predictions: list[str], references: list[str]) -> dict:
        entry_results = [
            self._engine.compute(predictions=[pred], references=[ref])["score"] / 100
            for pred, ref in zip(predictions, references)
        ]
        results = self._engine.compute(predictions=predictions, references=references)
        results["score"] = results["score"] / 100
        results["single"] = entry_results
        mean, margin, conf = compute_confidence_interval(entry_results)
        results["margin"] = margin
        results["conf"] = conf
        return results


class BLEURT(Metric):
    def __init__(self) -> None:
        self._engine = evaluate.load("bleurt", module_type="metric", checkpoint="BLEURT-20")

    def compute(self, predictions: list[str], references: list[str]) -> dict:
        results = self._engine.compute(predictions=predictions, references=references)
        results["mean_score"] = sum(results["scores"]) / len(results["scores"])
        mean, margin, conf = compute_confidence_interval(results["scores"])
        results["margin"] = margin
        results["conf"] = conf
        return results


class ROUGE(Metric):
    def __init__(self) -> None:
        self._engine = evaluate.load("rouge")

    def compute(self, predictions: list[str], references: list[str]) -> dict:
        entry_results = [
            self._engine.compute(predictions=[pred], references=[ref])["rougeL"]
            for pred, ref in zip(predictions, references)
        ]
        results = self._engine.compute(predictions=predictions, references=references)
        mean, margin, conf = compute_confidence_interval(entry_results)
        results["single_rougel"] = entry_results
        results["margin"] = margin
        results["conf"] = conf
        return results


class COMET(Metric):
    def __init__(self) -> None:
        self._engine = evaluate.load("comet")

    def compute(self, predictions: list[str], references: list[str], sources: list[str]) -> dict:
        results = self._engine.compute(predictions=predictions, references=references, sources=sources)
        mean, margin, conf = compute_confidence_interval(results["scores"])
        results["margin"] = margin
        results["conf"] = conf
        return results


class BERTSCORE(Metric):
    def __init__(self) -> None:
        self._engine = evaluate.load("bertscore")

    def compute(self, predictions: list[str], references: list[str]) -> dict:
        results = self._engine.compute(predictions=predictions, references=references, lang="pt")
        results["mean_f1"] = sum(results["f1"]) / len(results["f1"])
        return results


class VIdScore(Metric):
    """Variety identification score.
    The goal of the VId score is to measure the variety of the translations.
    When the variety of the prediction is classified close to the variety of the reference, the score is higher.
    Else, the score is lower.
    """

    MODEL_NAME = "u1537782/LVI_bert-large-portuguese-cased"

    def __init__(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)

    def compute(self, predictions: list[str], references: list[str]) -> dict:
        n = len(predictions)
        probs_pred = self._pipeline(predictions)
        pred_pt_score = probs_pred[:, 1]
        preds_pred = probs_pred.argmax(dim=1)
        n_pt_pred = (preds_pred == 0).sum()
        perc_pt_pred = n_pt_pred.item() / n

        probs_refs = self._pipeline(references)
        refs_pt_score = probs_refs[:, 1]
        preds_refs = probs_refs.argmax(dim=1)
        n_pt_refs = (preds_refs == 0).sum()
        perc_pt_refs = n_pt_refs.item() / n

        agreement = (preds_pred == preds_refs).sum().item() / n
        scores = (pred_pt_score - refs_pt_score).tolist()
        mean, margin, conf = compute_confidence_interval(scores)

        return {
            "perc_pt_pred": perc_pt_pred,
            "perc_pt_refs": perc_pt_refs,
            "agreement": agreement,
            "score": mean,
            "scores": scores,
            "margin": margin,
            "conf": conf,
        }

    def _pipeline(self, texts: list[str]) -> list[dict]:
        encoded_input = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            output = self._model(**encoded_input)
        logits = output.logits
        probs = torch.softmax(logits, dim=1)
        return probs
