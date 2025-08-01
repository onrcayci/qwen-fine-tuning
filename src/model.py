from statistics import mode

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Qwen2Tokenizer,
    Qwen3ForTokenClassification,
)
from transformers.modeling_outputs import TokenClassifierOutput

LABEL2ID = {
    "O": 0,
    "B-CARDINAL": 1,
    "B-DATE": 2,
    "I-DATE": 3,
    "B-PERSON": 4,
    "I-PERSON": 5,
    "B-NORP": 6,
    "B-GPE": 7,
    "I-GPE": 8,
    "B-LAW": 9,
    "I-LAW": 10,
    "B-ORG": 11,
    "I-ORG": 12,
    "B-PERCENT": 13,
    "I-PERCENT": 14,
    "B-ORDINAL": 15,
    "B-MONEY": 16,
    "I-MONEY": 17,
    "B-WORK_OF_ART": 18,
    "I-WORK_OF_ART": 19,
    "B-FAC": 20,
    "B-TIME": 21,
    "I-CARDINAL": 22,
    "B-LOC": 23,
    "B-QUANTITY": 24,
    "I-QUANTITY": 25,
    "I-NORP": 26,
    "I-LOC": 27,
    "B-PRODUCT": 28,
    "I-TIME": 29,
    "B-EVENT": 30,
    "I-EVENT": 31,
    "I-FAC": 32,
    "B-LANGUAGE": 33,
    "I-PRODUCT": 34,
    "I-ORDINAL": 35,
    "I-LANGUAGE": 36,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class QwenClassifier:

    def __init__(self) -> None:
        self.model: Qwen3ForTokenClassification = (
            AutoModelForTokenClassification.from_pretrained(
                "Qwen/Qwen3-1.7B",
                num_labels=len(LABEL2ID),
                label2id=LABEL2ID,
                id2label=ID2LABEL,
            )
        )
        self.tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-1.7B"
        )

    def __call__(self, data: list[str]) -> list[list[int]]:
        inputs = self.tokenizer(data, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs: TokenClassifierOutput = self.model(**inputs)

        preds = outputs.logits.argmax(dim=-1).tolist()
        predictions: list[list[int]] = []

        for i in range(len(data)):
            predictions.append(
                self._get_word_level_classification(inputs.word_ids(i), preds[i])
            )

        return predictions

    def _get_word_level_classification(
        self, word_ids: list[int], token_predictions: list[int]
    ) -> list[int]:
        predictions = []
        prev_word_id = -1
        curr_token_predictions = []

        for word_id, token_prediction in zip(word_ids, token_predictions):
            if word_id is None:
                continue

            # new word begins
            if prev_word_id != word_id:
                # save the previous word prediction
                if curr_token_predictions:
                    predictions.append(mode(curr_token_predictions))
                    curr_token_predictions = []

                # start the new word prediction
                curr_token_predictions.append(token_prediction)
                prev_word_id = word_id

            # continue the current word prediction
            else:
                curr_token_predictions.append(token_prediction)

        # check for the last word
        if curr_token_predictions:
            predictions.append(mode(curr_token_predictions))

        return predictions


if __name__ == "__main__":
    model = QwenClassifier()
    model(["This is", "Onurasda"])
