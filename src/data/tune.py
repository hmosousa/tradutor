"""Datasets for tuning models with torchtune."""

from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from torchtune.data import (
    CROSS_ENTROPY_IGNORE_IDX,
    InstructTemplate,
    Message,
    validate_messages,
)
from torchtune.datasets._instruct import InstructDataset
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.tokenizers import BaseTokenizer


class PTradutorTemplate(InstructTemplate):
    """
    Prompt template for preference datasets similar to StackExchangedPaired.
    """

    template = "{input}"

    @classmethod
    def format(cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None) -> str:
        """
        Generate prompt from instruction and input.

        Args:
            sample (Mapping[str, Any]): a single data sample with instruction
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                placeholder names in the template to the column names in the sample.
                If None, assume these are identical.

        Returns:
            The formatted prompt
        """
        column_map = column_map or {}
        key_input = column_map.get("input", "en")
        prompt = cls.template.format(input=sample[key_input])

        return prompt


class PTradutorDataset(InstructDataset):
    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        transformed_sample = self._transform(sample) if self._transform else sample

        prompt = self.template.format(transformed_sample, self._column_map)
        key_output = self._column_map["output"] if self._column_map and "output" in self._column_map else "output"
        messages = [
            Message(
                role="system",
                content="You are a translator from English to European Portuguese",
                masked=(not self.train_on_input),
            ),
            Message(
                role="user",
                content=f"Translate this text from English to European Portuguese: {prompt}",
            ),
            Message(role="assistant", content=transformed_sample[key_output]),
        ]

        validate_messages(messages)

        tokens, mask = self._tokenizer.tokenize_messages(messages)

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, tokens))
        assert len(tokens) == len(labels)

        return {"tokens": tokens, "labels": labels}


def ptradutor_dataset(
    tokenizer: BaseTokenizer,
    source: str = "u1537782/PTradutor",
    max_seq_len: int = 1024,
    train_on_input: bool = True,
    split: str = "train",
    clean: bool = False,
    packed: bool = False,
) -> PTradutorDataset:
    """
    Family of preference datasets similar to `StackExchangePaired data
    <https://huggingface.co/datasets/u1537782/PTradutor>`_.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data.
        source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
            Default is 1024.

    Returns:
        InstructDataset: The instruct dataset built from source paired data.
    """
    name = "superclean" if clean else "raw"

    ds = PTradutorDataset(
        tokenizer=tokenizer,
        source=source,
        template=PTradutorTemplate,
        column_map={
            "input": "en",
            "output": "pt",
        },
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        split=split,
        name=name,
    )
    return PackedDataset(ds, max_seq_len=max_seq_len) if packed else ds
