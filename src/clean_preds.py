import pandas as pd


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
    pred_files = ["/local2/amvepa91/RadFM/src/gli_run/brats_gli_3d_vqa_subjTrue_test_v3.json.test.csv",
                  "/local2/amvepa91/RadFM/src/met_run/brats_met_3d_vqa_subjTrue_test_v1.json.test.csv",
                  "/local2/amvepa91/RadFM/src/goat_run/brats_goat_3d_vqa_subjTrue_test_v1.json.test.csv"]
    for pred_file in pred_files:
        df = pd.read_csv(pred_file)
        df['Clean Pred'] = df['Pred'].apply(clean_prediction)
        new_file = pred_file.replace(".csv", "_clean.csv")
        df.to_csv(new_file, index=False)
