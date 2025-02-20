"""Datasets to be used for evaluation."""

from abc import ABC

import datasets


class EvaluationDataset(ABC):
    def __init__(self) -> None:
        self._name = None
        self._data = None

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for entry in self._data:
            yield entry["idx"], entry["en"], entry["pt"]

    def __getitem__(self, idx):
        entry = self._data[idx]
        return entry["idx"], entry["en"], entry["pt"]

    @property
    def target(self):
        return self._data["pt"]

    @property
    def source(self):
        return self._data["en"]

    @property
    def name(self):
        return self._name

    def batch(self, batch_size: int):
        for i in range(0, len(self), batch_size):
            yield list(range(i, i + batch_size)), self[i : i + batch_size]


class Publico:
    def __init__(self):
        self._name = "publico"
        self._source = datasets.load_dataset("u1537782/Publico", "en", split="test")
        self._target = datasets.load_dataset("u1537782/Publico", "pt", split="test")

    def __len__(self):
        return len(self._source)

    def __iter__(self):
        for src, tgt in zip(self._source, self._target):
            yield src["id"], src["text"], tgt["text"]

    def __getitem__(self, idx):
        return self._source[idx]["id"], self._source[idx]["text"], self._target[idx]["text"]

    @property
    def target(self):
        return self._target["text"]

    @property
    def source(self):
        return self._source["text"]

    @property
    def name(self):
        return self._name


class FRMT:
    def __init__(self):
        self._name = "frmt"
        dataset = datasets.load_dataset("u1537782/frmt", split="test")
        _frmt = dataset.filter(lambda x: x["pt"] is not None)
        _frmt = _frmt.remove_columns(["br", "tw", "cn", "tw_simplified"])
        _frmt = _frmt.to_pandas().drop_duplicates()
        self._frmt = datasets.Dataset.from_pandas(_frmt)

    def __len__(self):
        return len(self._frmt)

    def __iter__(self):
        for idx, entry in enumerate(self._frmt):
            yield idx, entry["en"], entry["pt"]

    def __getitem__(self, idx):
        entry = self._frmt[idx]
        return idx, entry["en"], entry["pt"]

    @property
    def target(self):
        return self._frmt["pt"]

    @property
    def source(self):
        return self._frmt["en"]

    @property
    def name(self):
        return self._name


class DSLTL:
    def __init__(self):
        self._name = "dsl_tl"
        dataset = datasets.load_dataset("u1537782/PTradutor", name="raw", split="test")
        self._dsl = dataset.filter(lambda x: x["source"] == "dsl_tl")

    def __len__(self):
        return len(self._dsl)

    def __iter__(self):
        for entry in self._dsl:
            yield entry["idx"], entry["en"], entry["pt"]

    def __getitem__(self, idx):
        entry = self._dsl[idx]
        return entry["idx"], entry["en"], entry["pt"]

    @property
    def target(self):
        return self._dsl["pt"]

    @property
    def source(self):
        return self._dsl["en"]

    @property
    def name(self):
        return self._name


class NTREX(EvaluationDataset):
    def __init__(self):
        self._name = "ntrex"
        pt = datasets.load_dataset("davidstap/NTREX", "por_Latn", split="test")["text"]
        en = datasets.load_dataset("davidstap/NTREX", "eng_Latn", split="test")["text"]
        ids = list(range(len(pt)))
        self._data = datasets.Dataset.from_dict({"idx": ids, "en": en, "pt": pt})
