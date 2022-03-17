# -*- coding: utf-8 -*-
"""Question_answering_(PyTorch)

Install the Transformers and Datasets libraries to run this notebook.
"""

# !pip install datasets transformers[sentencepiece]
# !pip install accelerate
# # To run the training on TPU, you will need to uncomment the followin line:
# # !pip install cloud-tpu-client==0.10 torch==1.9.0 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
# !apt install git-lfs

"""You will need to setup git, adapt your email and name in the following cell.

You will also need to be logged in to the Hugging Face Hub. Execute the following and enter your credentials.
"""

from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer,BertTokenizerFast
from transformers import AutoModelForQuestionAnswering, BertForQuestionAnswering
import collections
import numpy as np
from datasets import load_metric
from tqdm.auto import tqdm
from transformers import TrainingArguments
from transformers import Trainer
from transformers import pipeline
import torch

#notebook_login() #TODO enable it if we eant push to hub


train_path ="train.json"
dev_path ="validation.json"
 
data_files = {"train": train_path, "validation": dev_path}
raw_datasets = load_dataset("json", data_files=data_files,field='data')

raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1)

# model_checkpoint = "bert-base-cased"
# model_checkpoint = "onlplab/alephbert-base"
model_checkpoint = "avichr/heBERT"

tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)

context = raw_datasets["train"][0]["context"]
question = raw_datasets["train"][0]["question"]

inputs = tokenizer(question, context)
tokenizer.decode(inputs["input_ids"])

# small_train=raw_datasets["train"].shuffle(seed=42).select(range(80))
# small_val=raw_datasets["train"].shuffle(seed=42).select(range(25))
# train_set=small_train
# val_set=small_val

train_set=raw_datasets["train"]
val_set=raw_datasets["validation"]

max_length = 384
stride = 128


def preprocess_training_examples(examples):
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

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

train_dataset = train_set.map(
    preprocess_training_examples,
    batched=True,
    remove_columns=train_set.column_names,
)

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

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
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

args = TrainingArguments(
    "heBERT-finetuned-squad",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=15,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=False, #TODO enable it if we eant push to hub
) 

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)
trainer.train() #TODO enable it if we want to train


# Save the model
output_dir = "hebert-finetuned-hebrew-squad"
trainer.save_model(output_dir)

