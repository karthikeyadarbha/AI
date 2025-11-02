# Small unit test to verify preprocess returns labels as list-of-lists and pad ids replaced by -100
import sys
from transformers import AutoTokenizer
from train_prompt_tuning import preprocess

def main():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    # set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = {
        "sentence": ["This movie was great!", "I did not like the film."],
        "label": [1, 0]
    }
    out = preprocess(examples, tokenizer, max_input_length=32, max_target_length=4)
    labels = out["labels"]
    assert isinstance(labels, list), "labels should be a list"
    assert all(isinstance(l, list) for l in labels), "each label should be a list"
    # ensure pad ids replaced with -100
    for lab in labels:
        assert all((tok == -100) or isinstance(tok, int) for tok in lab), "label tokens must be ints or -100"
    print("test_preprocess PASSED")

if __name__ == "__main__":
    main()