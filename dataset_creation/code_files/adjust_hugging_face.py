import codecs
import json
import os
import argparse
from code_files.utils import make_copy_of_json
parser = argparse.ArgumentParser()
parser.add_argument('-dst','--dest_dir', help='Location of dest_dir translated directory', required=True)
parser.add_argument('-orig_path_file','--orig_path_file', help='Location of orig translated file  path', required=True)



def adjust_dataset_format(orig_path,output_path):
    with open(orig_path, encoding="utf8") as f:
        dataset = json.load(f)
    new_format_dataset=[]

    temp={'id': '', 'title': '', 'context': '', 'question': '', 'answers': {'text': [], 'answer_start': []}}
    for article in (dataset['data']):
        for index_paragraph, paragraph in enumerate(article['paragraphs']):
            context = paragraph['context']
            for index_qa, qa in enumerate(paragraph['qas']):
                temp = {'id':qa['id'], 'title': article['title'], 'context': context, 'question': qa['question'], 'answers': {'text': [], 'answer_start': []}}
                for index_answer, answer in enumerate(qa['answers']):
                    ans_start = answer['answer_start']
                    ans = answer['text']
                    temp['answers']['text'].append((ans))
                    temp['answers']['answer_start'].append((ans_start))
                new_format_dataset.append(temp)
    new_dataset = {"data": new_format_dataset}

    with codecs.open(output_path, 'wb', encoding='utf-8') as f:
        json.dump(new_dataset, f, ensure_ascii=False)



def main(args_dest_dir=None,args_orig_path_file=None,name_to_save=None ,use_old_name = False):

    if(args_orig_path_file is None and args_dest_dir is None ):
        args = parser.parse_args()
        args_orig_path_file = args.orig_path_file
        args_dest_dir = args.dest_dir

    try:
        output_path=make_copy_of_json(args_orig_path_file , args_dest_dir,name_to_save,use_old_name)
    except OSError:
        return

    adjust_dataset_format(args_orig_path_file, output_path)



if __name__ == "__main__":
    main()
