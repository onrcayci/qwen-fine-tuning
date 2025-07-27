import torch
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

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


def run_ner(data: list[str], model, tokenizer) -> list[int]:
    inputs = tokenizer(data, return_tensors="pt", truncation=True, padding=True)
    inputs.to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = outputs.logits.argmax(dim=-1)
    predictions.to("cpu")

    return predictions


def run_eval(model, tokenizer) -> None:
    ontonotes5_test = load_dataset("hgissbkh/ontonotes5", split="test")
    results = []
    for example in ontonotes5_test:
        preds = run_ner(example["tokens"], model, tokenizer)
        results.append({"predictions": preds, "true_labels": example["tags"]})


def main() -> None:
    model = AutoModelForTokenClassification.from_pretrained(
        "Qwen/Qwen3-1.7B",
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model.to("cuda")
    run_eval(model, tokenizer)


if __name__ == "__main__":
    main()
