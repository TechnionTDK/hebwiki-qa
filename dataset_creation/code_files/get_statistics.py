from __future__ import division
import json
import random

import argparse
from shutil import copyfile
import os
from pathlib import Path
import easygui as e
import webbrowser

from code_files.utils import is_hebrew

parser = argparse.ArgumentParser()
parser.add_argument('-t','--translated_path', help='Location of translated file path', required=True)


def in_range_to_statistics(index_title,index_paragraph,start_point, stop_point):
    [title_start,paragraph_start]=start_point
    [title_stop,paragraph_stop]=stop_point
    if not (title_start <= index_title and index_title <= title_stop):
        return False
    if  (index_title == title_start and paragraph_start >  index_paragraph):
        return False
    if (index_title == title_stop and index_paragraph >= paragraph_stop):
        return False
    return True


def get_statistics(translated_path,run_just_from_point_to_point=False,start_point=None,stop_point=None):

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
            if (run_just_from_point_to_point):
                if(not in_range_to_statistics(index_article,index_paragraph,start_point, stop_point)):
                    continue
            if (not is_hebrew(paragraph['context'])):
                paragraphs_not_in_HEBREW=paragraphs_not_in_HEBREW+1
                # return     [paragraphs_in_HEBREW,paragraphs_not_in_HEBREW,answers_not_in_context,answers_in_context,total_pars,total_quests,total_articles,total_ans ]#TODO REMOVE AFTER DEBUG, Now stop when not in HEBREW
                continue
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
                        print("number of paragraph " + str(total_pars))
                        print("number of question for this paragraph " + str(toatl_q_per_par))
                        print(answer)
                        answers_not_in_context = answers_not_in_context + 1
            toatl_q_per_par = 0

               # qa['question'] - #TODO to do something with this?

        total_articles += 1


    return [paragraphs_in_HEBREW, paragraphs_not_in_HEBREW, answers_not_in_context, answers_in_context, total_pars,
        total_quests, total_articles,total_ans]

def main(args_translated_path=None , start_point=None , stop_point=None ):

    if(args_translated_path is None ):
        args = parser.parse_args()
        args_translated_path = args.translated_path



    if (start_point is None and stop_point is None ):
        run_just_from_point_to_point = False
    else:
        run_just_from_point_to_point = True

    #check if the answer is in the context:
    [paragraphs_in_HEBREW, paragraphs_not_in_HEBREW, answers_not_in_context, answers_in_context, total_pars,
     total_quests, total_articles,total_ans]=get_statistics(args_translated_path,run_just_from_point_to_point,start_point,stop_point)
    print("######################################################")
    print("######################################################")
    print("name of file:" , args_translated_path )
    print("paragraphs in HEBREW: ", paragraphs_in_HEBREW)
    print("paragraphs not in HEBREW: ", paragraphs_not_in_HEBREW)
    print("paragraphs answers in context: ", answers_in_context)
    print("paragraphs answers not in context: ", answers_not_in_context)
    print("paragraphs answers not in context: ", answers_not_in_context)
    rate=0
    if total_quests != 0:
        rate=(answers_not_in_context/total_ans)
    print("Rate: ",rate)
    print("Number of articles: ", total_articles)
    print("Number of paragraphs: ", total_pars)
    print("Number of questions: ", total_quests)
    print("Number of answers: ", total_ans)
    print("######################################################")
    print("######################################################")
    print("Finished get statistics")

if __name__ == "__main__":
    main()
