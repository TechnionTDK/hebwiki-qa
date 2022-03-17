from __future__ import division
import json
import codecs
import os
import random
from google.cloud import translate_v2
import argparse
from shutil import copyfile
import os
from pathlib import Path
import easygui as e
import webbrowser
from code_files.utils import is_hebrew,make_copy_of_json


parser = argparse.ArgumentParser()
parser.add_argument('-c','--cred', help='Google Translate Credentials', required=True)
parser.add_argument('-s','--squad', help='Location of SQuAD to translate', required=True)
parser.add_argument('-t','--translated', help='Location of translated directory', required=True)
parser.add_argument('-b','--batch_size', help='batch size', required=False)
parser.add_argument('-name','--name_to_save', help='name to save', required=False)
parser.add_argument('-use_old_name','--use_old_name', help='use old name for saving', required=False)



def data_translate(filename,translate_client,batch_size):

    title_start=0
    context_start=0
    title_stop=0
    context_stop=0
    flag_for_start=0
    flag_for_stop=0

    total_pars = 0
    total_quests = 0
    total_articles = 0
    with open(filename, encoding="utf8") as f:
        dataset = json.load(f)
    for index_article,article in enumerate(dataset['data']):
        if (flag_for_stop == 0):
            print("Translating " + article['title'])
        for index_paragraph,paragraph in enumerate(article['paragraphs']):
            if total_quests >= batch_size:
                if (flag_for_stop == 0):
                    title_stop = index_article
                    context_stop = index_paragraph
                    flag_for_stop = 1
                break
            if (is_hebrew(paragraph['context'])):
                print("Paragraph already in HEBREW, skipping")
                continue
            else:
                print("Translating paragraph of " + article['title'])

            if (flag_for_start == 0):
                title_start = index_article
                context_start = index_paragraph
                flag_for_start = 1


            total_pars += 1
            to_print = random.randint(1, 6)  # don't always print
            to_print=0
            if (to_print == 1):
                print(paragraph['context'])
            for index_qa,qa in enumerate(paragraph['qas']):
                total_quests += 1
                for index_answer,answer in enumerate(qa['answers']):
                    text = answer['text'].replace('\"', '')
                    t = translate_text(text, translate_client)
                    answer['text'] = t
                question = qa['question'].replace('\"', '')
                t = translate_text(question, translate_client)
                qa['question'] = t

            context = paragraph['context'].replace('\"', '')
            t = translate_text(context, translate_client)
            paragraph['context'] = t
            if (to_print == 1):
                print("######################################################")
                print("Translated text")
                print("######################################################")
                print(paragraph['context'])

            # # save every paragraph:
            # with codecs.open(filename, 'wb', encoding='utf-8') as f:
            #     json.dump(dataset, f, ensure_ascii=False)

        total_articles += 1
        # # save every article:
        # if (flag_for_stop == 0):
        #     with codecs.open(filename, 'wb', encoding='utf-8') as f:
        #         json.dump(dataset, f, ensure_ascii=False)
    print("######################################################")
    print("######################################################")
    print("Translated so far:")
    print("Number of articles: ", total_articles)
    print("Number of paragraphs: ", total_pars)
    print("Number of questions: ", total_quests)
    print("######################################################")
    print("######################################################")


    with codecs.open(filename, 'wb', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False)

    print("Finished Translation squad")

    start_point = [title_start, context_start]
    stop_point = [title_stop, context_stop]
    return start_point, stop_point

def translate_text(text,translate_client):
    # The target language
    target = 'he'
    # Translates text into HEBREW using Neural Machine Translation
    translation = translate_client.translate(
        text,
        target_language=target,source_language='en'
    )

    translated_text = translation['translatedText']
    return translated_text


def main(args_squad=None,args_translated=None,args_cred=None,batch_size=None,name_to_save=None ,use_old_name = False):


    if(args_squad is None and args_cred is None and args_translated is None ):
        args = parser.parse_args()
        args_squad=args.squad
        args_translated=args.translated
        args_cred=args.cred

    try:
        output_translated_squad=make_copy_of_json(args_squad , args_translated,name_to_save,use_old_name)
    except OSError:
        return
    translate_client = translate_v2.Client.from_service_account_json(args_cred)
    start_point, stop_point = data_translate(output_translated_squad,translate_client,batch_size)
    return  start_point, stop_point

if __name__ == "__main__":
    main()
