import pandas as pd
import json


# -----------------------------------------------------
# 1) Define a helper function to "clean" predictions
# -----------------------------------------------------
def clean_prediction(predicted_text: str) -> str:
    """
    Removes repeated phrases if the entire prediction
    is just a repeated chunk.
    """
    # Remove surrounding brackets if present:
    predicted_text = predicted_text.strip("[']").strip()

    # Split into tokens:
    tokens = predicted_text.split()

    if not tokens:
        return ""  # Return empty if there's nothing

    # Attempt to detect the smallest repeated chunk
    length = len(tokens)
    for n in range(1, length + 1):
        chunk = tokens[:n]
        next_chunk = tokens[n:2*n]
        # if repeated pattern is found in the beginning, return initial chunk
        if chunk == next_chunk:
            return " ".join(chunk)
        # if repeated pattern is found after the beginning, return initial subchunk
        else:
            for j in range(1, n):
                sub_chunk = tokens[j:n]
                next_sub_chunk = tokens[n:2*n-j]
                if sub_chunk == next_sub_chunk:
                    return " ".join(chunk[:j])
    # If no repeated pattern is found, return the original text
    return predicted_text


if __name__ == "__main__":
    pred_gt_files = [("RadFM/src/goat_new_dataset_v3_run/brats_goat_3d_vqa_subjTrue_test_updated_v3_seed0_multitask_fixed.json.test.csv",
                      "brats_goat_3d_vqa_subjTrue_test_updated_v3_seed0_multitask_fixed.json")]
    for pred_file, gt_file in pred_gt_files:
        print("Cleaning predictions in", pred_file)
        df = pd.read_csv(pred_file)
        df['Clean Pred'] = df['Pred'].apply(clean_prediction)
        new_file = pred_file.replace(".csv", "_clean.csv")
        df.to_csv(new_file, index=False)
        with open(gt_file, "r") as f:
            gt_data = json.load(f)
        print("length of csv: ", len(df), "length of gt: ", len(gt_data))
        for (_, row), gt_datum in zip(df.iterrows(), gt_data):
            gt_datum['model_answer'] = row['Clean Pred']
        with open(pred_file.replace(".csv", ".json"), "w") as f:
            json.dump(gt_data, f, indent=4)

