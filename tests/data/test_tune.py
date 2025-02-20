# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# from unittest.mock import patch

import pytest
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules.tokenizers import SentencePieceBaseTokenizer

# from src.constants import ROOT
# from src.data.tune import ptradutor_dataset


class TestPTradutorDataset:
    @pytest.fixture
    def tokenizer(self):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        return SentencePieceBaseTokenizer(str(ROOT / "tests" / "assets" / "m.model"))


#     @patch("torchtune.datasets._instruct.load_dataset")
#     def test_label_no_masking(self, load_dataset, tokenizer):
#         """
#         Test whether the input and the labels are correctly created when the input is not masked.
#         """

#         # mock the call to HF datasets
#         load_dataset.return_value = [
#             {
#                 "en": "Give three tips for staying healthy.",
#                 "pt": "Dê três dicas para se manter saudável.",
#             }
#         ]

#         ds = ptradutor_dataset(tokenizer=tokenizer, train_on_input=True, packed=False)
#         input, labels = ds[0]["tokens"], ds[0]["labels"]

#         assert len(input) == len(labels)
#         assert labels[-1] == tokenizer.eos_id
#         assert input[0] == tokenizer.bos_id
#         assert CROSS_ENTROPY_IGNORE_IDX not in labels

#     def test_init(self, tokenizer):
#         ds = ptradutor_dataset(tokenizer)
#         assert len(ds) == 3_943_928
