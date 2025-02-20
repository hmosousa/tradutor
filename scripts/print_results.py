import json
import os
from fire import Fire

import pandas as pd

from src.constants import ROOT


def main(dataset: str = "frmt"):
    """dataset: "frmt" or "ntrex"""
    output_folder = ROOT / "results" / dataset
    json_files = [pos_json for pos_json in os.listdir(output_folder) if pos_json.endswith(".json")]
    test_output = pd.DataFrame(columns=["model_name", "bleu", "rougeL", "comet", "conf", "vid"])

    for index, js in enumerate(json_files):
        with open(os.path.join(output_folder, js)) as json_file:
            json_text = json.load(json_file)
            bleu = round(json_text["bleu"]["score"] * 100, 2)
            rougeL = round(json_text["rouge"]["rougeL"] * 100, 2)
            comet = round(json_text["comet"]["mean_score"] * 100, 2)
            conf = round(json_text["comet"]["margin"] * 100, 2)
            vid = round(json_text["vid_score"]["perc_pt_pred"] / json_text["vid_score"]["perc_pt_refs"], 3)
            model_name = js.replace(".json", "")
            test_output.loc[index] = [model_name, bleu, rougeL, comet, conf, vid]
    print(test_output)


if __name__ == "__main__":
    Fire(main)
