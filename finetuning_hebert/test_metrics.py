from datasets import load_dataset
from transformers import AutoTokenizer,BertTokenizerFast
from transformers import AutoModelForQuestionAnswering, BertForQuestionAnswering
import collections
import numpy as np
from datasets import load_metric
from tqdm.auto import tqdm
from transformers import Trainer




def compute_metrics_wrap(model_checkpoint,raw_datasets):

    tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
    val_set = raw_datasets["validation"]
    max_length = 384
    stride = 128

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    validation_dataset = val_set.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=val_set.column_names,
    )

    metric = load_metric("squad")

    n_best = 20
    max_answer_length = 30

    def compute_metrics(start_logits, end_logits, features, examples):
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
                end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                                end_index < start_index
                                or end_index - start_index + 1 > max_answer_length
                                or offsets[end_index] == []
                                or offsets[start_index] == []
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0]: offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers)

    model = BertForQuestionAnswering.from_pretrained(model_checkpoint)

    trainer = Trainer(
        model=model,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )

    predictions = trainer.predict(validation_dataset)
    (start_logits, end_logits) = predictions[0][0], predictions[0][1]

    return compute_metrics(start_logits, end_logits, validation_dataset, val_set)


if __name__ == "__main__":

    # Please choose model:

    # model_checkpoint = "bert-base-cased"
    # model_checkpoint = "onlplab/alephbert-base"
    # model_checkpoint = "avichr/heBERT"
    # model_checkpoint = "our_hebert-finetuned-squad"
    model_checkpoint = "hebert-finetuned-hebrew-squad"


    # Please choose dataset:
    # raw_datasets = load_dataset("squad")
    train_path = "train.json"
    dev_path = "validation.json"
    data_files = {"train": train_path, "validation": dev_path}
    raw_datasets = load_dataset("json", data_files=data_files, field='data')

    result = compute_metrics_wrap(model_checkpoint,raw_datasets)

    print('############################################')
    print(result)
    print('############################################')