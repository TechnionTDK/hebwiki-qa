# hebwiki-qa
### About

NLP is a field of linguistics and machine learning focused on understanding everything related to human language. The aim of NLP tasks is not only to understand single words individually, but to be able to understand the context of those words. A common task in this fields is **Extracting an answer from a text**: Given a question and a context, extracting the answer to the question based on the information provided in the context.
In this project, our goal was to create a a question-answering model for the Hebrew language. For this purpose, we performed fine-tuning on heBERT model with a translated SQUAD dataset that we have created. This task comes in many flavors, but we focused on extractive question answering. This involves posing questions about a document and identifying the answers as spans of text in the document itself.

## Dataset Creation
A common Dataset for question-answering task is SQUAD. Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. The first step in our project was to create a proper SQUAD dataset in Hebrew. 
 
### Dataset Structure
The SQUAD dataset is a json file with an hierarchy of topics that contain paragraphs, where each paragraph contains contexts, and each context contains questions and answers.

An element in the dataset has this structure:

```
{
"answers": [{
  "answer_start": 403,
  "text": "Santa Clara, California"
}, {
  "answer_start": 355,
  "text": "Levi's Stadium"
  }, {
  "answer_start": 355,
  "text": "Levi's Stadium in the San Francisco Bay Area at Santa Clara, California."
  }
],
"question": "Where did Super Bowl 50 take place?",
"id": "56be4db0acb8001400a502ee"
}
```

After translating and processing:
```
{
 "answers": [{
  "answer_start": 311,
  "text": "סנטה קלרה, קליפורניה"
  }, {
    "answer_start": 271,
    "text": "אצטדיון ליווי"
     }
],"question": "היכן התקיים סופרבול 50?",
 id": "56be4db0acb8001400a502ee"
 }
```
We can see that after translation, the "answer_start" changed to the place that match the translated context. 
Note, that after tranlsatio there is only two answers for the quiestion, while in the original dataset there are three answers.
In the next section we will explain how we generated the final Hebrew SQUAD. 

notes:
* The original SQUAD dataset files was taken from the official site: https://rajpurkar.github.io/SQuAD-explorer/
* for our implementation we used "dataset creation" part of SOQAL project: https://github.com/husseinmozannar/SOQAL/tree/master/dataset_creation/

### Dataset Translation and Process
#### Translation
The first stage of dataset creation process was translating SQUAD dataset from English to Hebrew.
We used google translation API.
This part is implemented in translate_squad.py

#### Fix Translation

A question-answer structure contains information about the location of the answer in the context.
Since the translation changes the context, the start location also need to be updated.
To find what is the new place of the answer in Hebrew, we took the original answer's location, 
We searched the translated answer in the translated text three times: once in the whole context once from the beginning of the text to the original location, and once from the original location to the end of the context. We calculated the distance between the location we found each time to the original location.
We took the place the minimize the distance.
In some cases, although the original context contained the original answer, after translation the answer does not appear exactly in the paragraph.
If the answer was not found in the translated text, we put "-1" in the "answer_start" field to mark it.
in this part we also handled some special characters as punctuation and quotes signs that was not uniformly translated and  

This part is implemented in fix_translate.py.

#### Remove bad Translations

As described in the previous section, in some cases the translated answer was not found in the translated context and marked with "-1".
in this part we cleaned the dataset from these. we made here an extra search of the answer in the context to verify. if the answer was not found or the location already marked with -1, we removed the answer. If all the answers of a question were deleted, we removed the question. If all the questions of a context were deleted, we removed the context. 

For example, if we look on the example above, the third answer: 
```
{
"answer_start":-1, 
"text": "אצטדיון ליווי באזור מפרץ סן פרנסיסקו בסנטה קלרה, קליפורניה."
}
```
This does not an exact match to the text in the context:
```
"context": "...באצטדיון ליווי'ס באזור מפרץ סן פרנסיסקו בסנטה קלרה, קליפורניה..."

```

Another case of inconsistency is when it comes to translating names or complex phrases:

In the original:\
Question: "Which NFL team represented the NFC at Super Bowl 50?"\
Answer: "Carolina Panthers"\
Context:\
"The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion **Carolina Panthers** 24-10 to earn their third Super Bowl title."

While in the tranlation:\
Question: "איזו קבוצת NFL ייצגה את ה-NFC בסופרבול 50"\
Answer:  "קרולינה פנתרים"\
Context: \
"...אלופת ועידת הכדורגל האמריקאית (AFC) דנבר ברונקוס ניצחה את אלופת ועידת הכדורגל הלאומית (NFC) **קרולינה פנתרס** 24–10 כדי לזכות בתואר הסופרבול השלישי שלה..."

This is not an exact match between the answer and the context, so the answer has been removed from the Hebrew dataset.


This part is implemented in remove_bad_tranlations.py.


#### Adjust To HuggingFace Format
The dataset format that is used by huggingface in the fine-tuning process (as described below) is a bit different that the original dataset structure.
Now, in each json dataset's sample, it contains context, question, and answers (instead of the previous hierarchy where the context is written once and contains all its questions). The new dataset is heavier, but it is necessary for using huggingface transformer's API.
```
{
            "id": "56be4db0acb8001400a502ee",
            "title": "Super_Bowl_50",
            "context": "סופרבול 50 היה משחק כדורגל אמריקאי כדי לקבוע את אלופת ליגת הפוטבול הלאומית (NFL) לעונת 2015. אלופת ועידת הכדורגל האמריקאית (AFC) דנבר ברונקוס ניצחה את אלופת ועידת הכדורגל הלאומית (NFC) קרולינה פנתרס 24–10 כדי לזכות בתואר הסופרבול השלישי שלה. המשחק נערך ב-7 בפברואר 2016 באצטדיון ליווי'ס באזור מפרץ סן פרנסיסקו בסנטה קלרה, קליפורניה. מכיוון שזה היה הסופרבול ה-50, הליגה הדגישה את יום השנה הזהב עם יוזמות שונות בנושא זהב, כמו גם השעיה זמנית את המסורת של שם כל משחק סופרבול עם ספרות רומיות (שתחתן המשחק היה ידוע בתור סופרבול L ), כך שהלוגו יוכל להציג באופן בולט את הספרות הערביות 50.",
            "question": "היכן התקיים סופרבול 50?",
            "answers": {
                "text": ["סנטה קלרה, קליפורניה", "אצטדיון ליווי"],
                "answer_start": [311, 271]
            }
}
```
### Statistics 
Since not all the translation succeeded, the amount of translated question-answer pairs in different from the original. we compared them to evaluate our dataset and make sure it is large enough for training task.

####  Develop set 
<!-- TABLE_GENERATE_START -->

|     | Original SQUAD  | Translated SQUAD |
| ------------- | ------------- | ------------- |
| Contexts  | 2,067  | 2,040  |
| Questions  | 10,570  | 7,455  |
| Answers  | 34,762  | 20,485  |

<!-- TABLE_GENERATE_END -->

####  Training set 
<!-- TABLE_GENERATE_START -->

|     | Original SQUAD  | Translated SQUAD |
| ------------- | ------------- | ------------- |
| Contexts  | 18,896  | 18,064  |
| Questions  | 87,599  | 52,405  |
| Answers  | 87,599  | 52,405  |

<!-- TABLE_GENERATE_END -->


The translation success rate is: 58.92% for the develop set, 59.82% for the training set.
This part is implemented in get_statistics.py.

### Download and Running  

#### Environment info
- Platform: Windows (Pycharm)
- Python version: 3.6

#### Environment setup
1. Clone the project's repository with the command:\
   ```git clone https://github.com/TechnionTDK/hebwiki-qa```
2. Create a virtual environment:\
   ```python -m venv hebwiki_env```
3. Activate the virtual environment:\
   ```source hebwiki_env/bin/activate```
4. Install the required packages:\
   ```python -m pip install -r dataset_creation/installs/requirements.txt```

#### Running the Project
1. First download the SQuAD datasets: train https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json, and dev https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json 
2. Obtain your own credentials for the Google Translate API: https://console.cloud.google.com/apis/credentials?project=hale-silicon-331517.
3. To translate and fix the dataset, run main.py located in dataset_creation directory:
```shell
python main.py ^
--work_dir DIRECTORY ^
--cred_file CREDENTIALS_FILE ^ 
--translate True/False ^
--squad_dev SQUAD_DEV_FILE ^ 
--squad_train SQUAD_TRAIN_FILE
```

--translate is a flag: True for translate with google api from squad or False if you just want to fix dataset 

For example:

```shell
python main.py --work_dir data_files --cred_file data_files/CREDENTIALS_DIRECTORY/foerl-colohn-396819-878daf965ecb.json --translate True --squad_dev data_files/SQUAD_DIRECTORY/dev-v1.1.json --squad_train data_files/SQUAD_DIRECTORY/train-v1.1.json
```

The script will create new .json datasets (train.json , validation.json) in HUGGING_FACE_FORMAT_TRANSLATED_DIRECTORY located in work_dir.

## Fine-Tuning heBERT-QA
### Implementation
#### Steps:
our implementation is based on Huggingface tutorial: https://huggingface.co/course/chapter7/7?fw=pt.
Here we describe the main changes and adjustment we did for our project.

##### Loading Dataset:

In the first step we load our dataset that saved locally. 
```
data_files = {"train": train_path, "validation": dev_path}
raw_datasets = load_dataset("json", data_files=data_files,field='data')
```

where train_path and dev_path is where our translated dataset is saved. 

The raw dataset was splitted into two parts:

```
train_set=raw_datasets["train"]
val_set=raw_datasets["validation"]
```


##### Loading Model And Tokenizer:

In this step we load a pre-trained model from Huggingface hub. we chose heBERT. HeBERT is a Hebrew pretrained language model. It is based on Google's BERT architecture and it is BERT-Base config. link to the model in huggingface hub: https://huggingface.co/avichr/heBERT



```
model_checkpoint = "avichr/heBERT"
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
model = BertForQuestionAnswering.from_pretrained(model_checkpoint)
```

##### Choosing trainig arguments

We tried differents parameters. Such as: learning_rate from 1e-5 to 3e-5, num_train_epochs from 1 to 15 and more. 
We chose the best parameters that maximize our performance on the validation dataset.   

```
args = TrainingArguments(
    "hebert-finetuned-hebrew-squad",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=15,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=False, 
)  
```

##### Training

```
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)
trainer.train() 
```
##### Saving Model

After training ends, we want to save the new model for evaluation and further use.
```
trainer.save_model(output_dir)
```

##### Evaluating Model 
To evaluate the trained models, we used test_metric.py.
Here, we have to load the trained model that we want to evaluate, and also load again the dataset (because we want to take the validation set).
We used the same metric that been used in the tutorial:

```
metric = load_metric("squad")
```
Then, to calculate the result we called:

```
model_checkpoint = "our_hebert-finetuned-squad"
result = compute_metrics_wrap(model_checkpoint,raw_datasets)
```


##### Push To Hub
To push the trained model to Huggingface hub, open push_to_hub.ipynb in Google Colab, upload the model and datssets and run.
we used the command:
```
 trainer.push_to_hub()
```

### Download and Running  

#### Environment info
- Platform: Linux
- Python version: 3.6.9

#### Environment setup
1. Clone the project's repository with the command: \
   ```git clone https://github.com/TechnionTDK/hebwiki-qa```
2. Create a virtual environment:\
   ```python -m venv hebwiki_env```
3. Activate the virtual environment:\
   ```source hebwiki_env/bin/activate```
4. Install the required packages:\
```pip install transformers```\
```pip install datasets transformers[sentencepiece]```\
```python -m pip install -r finetuning_hebert/requirements.txt```
   
#### Running the Project
1. Open new work directory \
  ```mkdir DIRECTORY``` \
  ```cd DIRECTORY```
2. Copy dataset files: train.json and dev.json to DIRECTORY. 
3. Run scripts: 
 - To fine-tune run the command ```screen python3 train_algorithm.py```
 - To test the fine-tuned model results run ```python3 test_metrics.py```
 - To push model to hub run ```push_to_hub.ipynb```notebook.
 - To show a usage example in Hebrew ```python3 example.py```



 ### Results 
+ The F1 score is the harmonic mean of the precision and recall. It can be computed with: F1 = 2 * (precision * recall) / (precision + recall). 
+ Our model achieved ‘f1 score ‘= 55.896% and ‘exact_match score’= 42.602% on validation dataset 
 ### Usage 
 In file example.py there is an example of usage of the Hebrew model.
First, load the model with pipeline:
 ```
from transformers import pipeline
output_dir = "our_hebert-finetuned-squad"
model_checkpoint=output_dir
question_answerer = pipeline("question-answering", model=model_checkpoint)
```
 
 The model can get also text that has been originally written in Hebrew. 

 Example on question and context from Hebrew Wikipedia:

```
context = "כרמלית היא כלי תחבורה ציבורית תת-קרקעי, היחיד בישראל. הכרמלית מחברת בין שלושה אזורים מרכזיים בעיר חיפה: העיר התחתית, שכונת הדר ומרכז הכרמל. לכרמלית קו בודד ובו שש תחנות פעילות, היא מופעלת על ידי חברת הכרמלית חיפה בעמ. הקמתה של הכרמלית החלה במאי 1956 והסתיימה במרץ 1959. בניגוד לתפיסה הרווחת, לפיה הכרמלית היא רכבת תחתית, אין היא אלא פוניקולר, רכבל הנע על מסילה במקום להיות תלוי באוויר. שלא כמו רכבת, אין בקרונות הכרמלית מנוע, ומשקלם של הקרונות היורדים הוא הכוח העיקרי המניע את הקרונות העולים (מנוע בתחנת הקצה העליונה תורם אף הוא כוח הנעה)."
question = "כמה תחנות יש בכרמלית?
 ```
 
 The result from the model:
 
 ```
 ans = question_answerer(question=question, context=context)
 {'score': 0.9899768829345703, 'start': 160, 'end': 162, 'answer': 'שש'}
```

Using the model in Hugging face hub:
  ![WhatsApp Image 2022-03-15 at 11 15 41](https://user-images.githubusercontent.com/50171760/158345526-cd3c0a1e-5e35-42c3-80f0-91e44ea1d844.jpeg)

## Huggingface Links
### Hebrew SQuAD
https://huggingface.co/datasets/tdklab/Hebrew_Squad_v1.1
### Fine-Tuned Model
https://huggingface.co/tdklab/hebert-finetuned-hebrew-squad

 ## Further Work
 Further development of the project can achieve improvement of the results and a better model of question-answering in Hebrew. It can be in some directions: 
+ Improve and increase the number of samples in the dataset: that includes fixing places where the translate did not performed well and adding question-answer pairs that originally written in Hebrew and not translated.
+ Examine different parameters of training as number of epochs, learning rate etc.
+ Examine more pre-trained Hebrew models. Currently, AlephBERT does not support our mission due to incompatibility of some parameters.  
 


## About Us
Created by Matan Ben-chorin, May Flaster, Guided by Dr. Oren Mishali. This is our final project as part of computer engineering B.Sc studies in the Faculty of Electrical Engineering combined with Computer Science at Technion, Israel Institute of Technology. For more cooperation, please contact email: Matan Ben-chorin: matan.bh1@gmail.com May Flaster: mayflaster96@gmail.com
