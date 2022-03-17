from __future__ import division
import json
import codecs
import argparse
import os

import easygui as e
from code_files.utils import is_hebrew,make_copy_of_json
parser = argparse.ArgumentParser()
parser.add_argument('-t','--fix_and_clean_translated_dir', help='Location of fix_and_clean translated directory', required=True)
parser.add_argument('-tf','--before_fix_translated_file', help='Location of translated file (before fix) path', required=True)

from itertools import filterfalse


def remove_bad_tranlations(filename):
    total_pars = 0
    total_quests = 0
    total_articles = 0
    total_qas_removed =0
    total_ans_removed =0
    total_paragraph_removed =0
    with open(filename, encoding="utf8") as f:
        dataset = json.load(f)
        new_dataset = dataset
    for article in (dataset['data']):
        total_articles += 1
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            if (not is_hebrew(paragraph['context'])):
                continue
            total_pars += 1
            for qa in paragraph['qas']:
                total_quests += 1
                len_before = len(qa['answers'])
                qa['answers'][:] = [x for x in qa['answers'] if (context.find(x['text']) >= 0 and x['answer_start'] != None)]
                len_after = len(qa['answers'])
                total_ans_removed= total_ans_removed+(len_before-len_after)

            len_before = len(paragraph['qas'])
            paragraph['qas'][:] = [x for x in paragraph['qas'] if not len(x['answers']) == 0]
            len_after = len(paragraph['qas'])
            total_qas_removed = total_qas_removed + (len_before - len_after)
        len_before = len(article['paragraphs'])
        article['paragraphs'][:] = [x for x in article['paragraphs'] if not len(x['qas']) == 0]
        len_after = len(article['paragraphs'])
        total_paragraph_removed = total_paragraph_removed + (len_before - len_after)
           # len(paragraph['qas']) == 0:
           #     article['paragraphs'].remove(paragraph)
           #     total_paragraph_removed += 1

    print("answers removed : ", total_ans_removed)
    print("question removed: ", total_qas_removed)
    print("paragraph removed: ", total_paragraph_removed)
    with codecs.open(filename, 'wb', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False)


def count_remaing(translated_path):

    with open(translated_path, encoding="utf8") as f:
        dataset = json.load(f)

    paragraphs_in_HEBREW=0
    paragraphs_not_in_HEBREW=0
    answers_not_in_context=0
    answers_in_context=0
    total_pars = 0
    total_quests = 0
    total_articles = 0
    total_ans=0
    toatl_q_per_par = 0
    for index_article,article in enumerate(dataset['data']):
        for index_paragraph,paragraph in enumerate(article['paragraphs']):
            if (not is_hebrew(paragraph['context'])):
                paragraphs_not_in_HEBREW=paragraphs_not_in_HEBREW+1
                # return     [paragraphs_in_HEBREW,paragraphs_not_in_HEBREW,answers_not_in_context,answers_in_context,total_pars,total_quests,total_articles,total_ans ]#TODO REMOVE AFTER DEBUG, Now stop when not in HEBREW
            else:
                paragraphs_in_HEBREW=paragraphs_in_HEBREW+1
            total_pars += 1
            context=paragraph['context']
            for index_qa,qa in enumerate(paragraph['qas']):
                total_quests += 1
                toatl_q_per_par +=1
                for index_answer,answer in enumerate(qa['answers']):
                    answer=answer['text']
                    total_ans=total_ans+1
                    if context.find(answer) >= 0:
                        answers_in_context = answers_in_context + 1
                        #print (answer)
                    else:
                        answers_not_in_context = answers_not_in_context + 1
            toatl_q_per_par = 0
        total_articles += 1
    print("######################################################")
    print("######################################################")
    print("paragraphs in HEBREW: ", paragraphs_in_HEBREW)
    print("paragraphs not in HEBREW: ", paragraphs_not_in_HEBREW)
    print("paragraphs answers in context: ", answers_in_context)
    print("paragraphs answers not in context: ", answers_not_in_context)
    print("paragraphs answers not in context: ", answers_not_in_context)
    rate = 0
    if total_quests != 0:
        rate = (answers_not_in_context / total_ans)
    print("Rate: ", rate)
    print("Number of articles: ", total_articles)
    print("Number of paragraphs: ", total_pars)
    print("Number of questions: ", total_quests)
    print("Number of answers: ", total_ans)
    print("######################################################")
    print("######################################################")
    print("Finished remove bad translations ")


def main(args_fix_and_clean_translated_dir=None,args_before_fix_translated_file=None,name_to_save=None ,use_old_name = False):

    if(args_before_fix_translated_file is None and args_fix_and_clean_translated_dir is None ):
        args = parser.parse_args()
        args_before_fix_translated_file = args.before_fix_translated_file
        args_fix_and_clean_translated_dir = args.fix_translated_dir

    try:
        output_fix_translated=make_copy_of_json(args_before_fix_translated_file , args_fix_and_clean_translated_dir,name_to_save,use_old_name)
    except OSError:
        return
    remove_bad_tranlations(output_fix_translated)
    count_remaing(os.path.join(args_fix_and_clean_translated_dir,name_to_save))


if __name__ == "__main__":
    main()
