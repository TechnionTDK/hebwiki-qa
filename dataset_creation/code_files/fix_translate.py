from __future__ import division
import json
import codecs
import argparse
import easygui as e
from code_files.utils import is_hebrew,make_copy_of_json
parser = argparse.ArgumentParser()
parser.add_argument('-t','--fix_translated_dir', help='Location of fix translated directory', required=True)
parser.add_argument('-tf','--before_fix_translated_file', help='Location of translated file (before fix) path', required=True)

def find_new_ans_start(ans_start_orig,context,answer):
    after_place=context.find(answer,ans_start_orig)
    before_place=context.rfind(answer,0,ans_start_orig)
    if before_place == -1 and after_place == -1 :
        return context.find(answer)
        # e.msgbox("Translated answer isn't exist in context", "Error")
        #return -1
    if before_place == -1  :
        if before_place > after_place:
            e.msgbox("Can't find new ans_start 2", "Error")
            return -1
        else:
            if context[after_place:(after_place+len(answer))] != answer:
                e.msgbox("Can't find new ans_start 3", "Error")
                return -1
            else:
                return after_place
    if after_place == -1  :
        if context[before_place:(before_place+len(answer))] != answer:
            e.msgbox("Can't find new ans_start 5", "Error")
            return -1
        else:
            return before_place




def fix_answer_start_num(filename):
    total_pars = 0
    total_quests = 0
    total_articles = 0
    with open(filename, encoding="utf8") as f:
        dataset = json.load(f)
    for article in dataset['data']:
        total_articles += 1
        for paragraph in article['paragraphs']:
            if (not is_hebrew(paragraph['context'])):
                continue
            total_pars += 1
            for qa in paragraph['qas']:
                total_quests += 1
                for answer in qa['answers']:
                    answer['answer_start'] = find_new_ans_start(answer['answer_start'],paragraph['context'],answer['text'])

    with codecs.open(filename, 'wb', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False)

def remove_niqqud_from_string(my_string):
    return ''.join(['' if  1456 <= ord(c) <= 1479 else c for c in my_string])

def remove_punctuation(filename):
    total_pars = 0
    total_quests = 0
    total_articles = 0
    with open(filename, encoding="utf8") as f:
        dataset = json.load(f)
    for article in dataset['data']:
        total_articles += 1
        for paragraph in article['paragraphs']:
            if (not is_hebrew(paragraph['context'])):
                continue
            total_pars += 1
            for qa in paragraph['qas']:
                total_quests += 1
                for answer in qa['answers']:
                    answer['text'] = remove_niqqud_from_string(answer['text'])
                qa['question'] = remove_niqqud_from_string(qa['question'])
            paragraph['context'] = remove_niqqud_from_string(paragraph['context'])


    with codecs.open(filename, 'wb', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False)

def repalce_bad_characters_from_string(my_string):
    new_string = my_string.replace('&#39;', '\'')
    new_string = new_string.replace('&quot;', '\"')
    return new_string

def repalce_bad_characters(filename):
    total_pars = 0
    total_quests = 0
    total_articles = 0
    with open(filename, encoding="utf8") as f:
        dataset = json.load(f)
    for article in dataset['data']:
        total_articles += 1
        for paragraph in article['paragraphs']:
            if (not is_hebrew(paragraph['context'])):
                continue
            total_pars += 1
            for qa in paragraph['qas']:
                total_quests += 1
                for answer in qa['answers']:
                    answer['text'] = repalce_bad_characters_from_string(answer['text'])
                qa['question'] = repalce_bad_characters_from_string(qa['question'])
            paragraph['context'] = repalce_bad_characters_from_string(paragraph['context'])


    with codecs.open(filename, 'wb', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False)


def main(args_fix_translated_dir=None,args_before_fix_translated_file=None,name_to_save=None ,use_old_name = False):

    if(args_before_fix_translated_file is None and args_fix_translated_dir is None ):
        args = parser.parse_args()
        args_before_fix_translated_file = args.before_fix_translated_file
        args_fix_translated_dir = args.fix_translated_dir

    try:
        output_fix_translated=make_copy_of_json(args_before_fix_translated_file , args_fix_translated_dir,name_to_save,use_old_name)
    except OSError:
        return


    remove_punctuation(output_fix_translated)
    repalce_bad_characters(output_fix_translated)
    fix_answer_start_num(output_fix_translated)





    print("Finished Fix translate")


if __name__ == "__main__":
    main()
