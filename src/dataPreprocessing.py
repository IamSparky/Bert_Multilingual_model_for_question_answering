import pandas as pd
import numpy as np
from config import config

def prepare_train_features(args, example):
    tokenizer = args.TOKENIZER
    
    tokenized_example = tokenizer(
        example["question"].lstrip(),
        example["context"].lstrip(),
        truncation="only_second",
        max_length=args.MAX_LEN,
        stride=args.STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = tokenized_example.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_example.pop("offset_mapping")

    for i, offsets in enumerate(offset_mapping):
        feature = {}

        input_ids = tokenized_example["input_ids"][i]
        attention_mask = tokenized_example["attention_mask"][i]
        token_type_ids = tokenized_example["token_type_ids"][i]

        feature['context'] = example["context"]
        feature['question'] = example["question"]
        feature['answer'] = example["answer_text"]
        feature['ids'] = input_ids
        feature['mask'] = attention_mask
        feature['token_type_ids'] = token_type_ids
        feature['offsets'] = offsets
        feature['padding_len'] = args.MAX_LEN

        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_example.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = example["answer_text"][sample_index]

        if example["answer_start"] == 0:
            feature["targets_start"] = cls_index
            feature["targets_end"] = cls_index
        else:
            start_char = example["answer_start"]
            end_char = start_char + len(example["answer_text"])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                feature["targets_start"] = cls_index
                feature["targets_end"] = cls_index
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                feature["targets_start"] = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                feature["targets_end"] = token_end_index + 1

    return feature

if __name__ == "__main__":
    df_train = pd.read_csv(config.TRAINING_FILE)
    index = 0
    df_train = df_train.iloc[index, :]

    output = prepare_train_features(
        args = config(),
        example = df_train
        )["tokens"]
                
    print(output)