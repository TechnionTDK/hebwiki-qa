from code_files.translate_squad import main as translate_squad
from code_files.fix_translate import main as fix_translate
from code_files.get_statistics import main as get_statistics
from code_files.remove_bad_tranlations import main as remove_bad_translations
from code_files.adjust_hugging_face import main as adjust_dataset_format

import os
import argparse

def translate_dev_and_fix_per_batch(work_dir,args_squad,cred_file):


    ###########################################  batch1  #############################################################
    batch_size = 1  # 1 is just for fast debug , you can write here 5000 for example
    name_to_save_after_translate = 'batch_1_dev.json'
    use_old_name = False
    dir_translated =  os.path.join(work_dir,'TRANSLATED_DIRECTORY')
    start_point, stop_point =translate_squad(batch_size=batch_size,use_old_name = use_old_name, name_to_save=name_to_save_after_translate, args_squad=args_squad,args_translated=dir_translated,args_cred=cred_file)

    args_fix_translated_dir =  os.path.join(work_dir,'FIX_TRANSLATED_DIRECTORY')
    args_before_fix_translated_file =os.path.join(dir_translated,name_to_save_after_translate)
    fix_translate( args_fix_translated_dir=args_fix_translated_dir , args_before_fix_translated_file=args_before_fix_translated_file,name_to_save=name_to_save_after_translate)

    args_translated_path=os.path.join(args_fix_translated_dir,name_to_save_after_translate)
    get_statistics(start_point=start_point, stop_point=stop_point ,args_translated_path=args_translated_path)



    ############################################  batch2  #############################################################
    batch_size = 1 # 1 is just for fast debug , you can write here 5000 for example
    name_to_save_after_translate = 'batch_2_dev.json'
    use_old_name = False
    args_squad = os.path.join(work_dir,'FIX_TRANSLATED_DIRECTORY','batch_1_dev.json')
    dir_translated =os.path.join(work_dir,'TRANSLATED_DIRECTORY')
    start_point, stop_point =translate_squad(batch_size=batch_size,use_old_name = use_old_name, name_to_save=name_to_save_after_translate, args_squad=args_squad,args_translated=dir_translated,args_cred=cred_file)

    args_fix_translated_dir = os.path.join(work_dir,'FIX_TRANSLATED_DIRECTORY')
    args_before_fix_translated_file =os.path.join(dir_translated,name_to_save_after_translate)
    fix_translate( args_fix_translated_dir=args_fix_translated_dir , args_before_fix_translated_file=args_before_fix_translated_file,name_to_save=name_to_save_after_translate)

    args_translated_path=os.path.join(args_fix_translated_dir,name_to_save_after_translate)
    get_statistics(start_point=start_point, stop_point=stop_point ,args_translated_path=args_translated_path)

    ############################################  batch3  #############################################################
    batch_size = 1 # 1 is just for fast debug , you can write here 100000 for example
    name_to_save_after_translate = 'batch_3_dev.json'
    use_old_name = False
    args_squad = os.path.join(work_dir,'FIX_TRANSLATED_DIRECTORY','batch_2_dev.json')
    dir_translated = os.path.join(work_dir,'TRANSLATED_DIRECTORY')
    start_point, stop_point = translate_squad(batch_size=batch_size, use_old_name=use_old_name,
                                              name_to_save=name_to_save_after_translate, args_squad=args_squad,
                                              args_translated=dir_translated, args_cred=cred_file)

    args_fix_translated_dir = os.path.join(work_dir,'FIX_TRANSLATED_DIRECTORY')
    args_before_fix_translated_file = os.path.join(dir_translated, name_to_save_after_translate)
    fix_translate(args_fix_translated_dir=args_fix_translated_dir,
                  args_before_fix_translated_file=args_before_fix_translated_file,
                  name_to_save=name_to_save_after_translate)

    args_translated_path = os.path.join(args_fix_translated_dir, name_to_save_after_translate)
    get_statistics(start_point=start_point, stop_point=stop_point, args_translated_path=args_translated_path)


def translate_train_and_fix_per_batch(work_dir,args_squad,cred_file):

    ###########################################  batch1  #############################################################
    batch_size = 1 # 1 is just for fast debug , you can write here 5000 for example
    name_to_save_after_translate = 'batch_1_train.json'
    use_old_name = False
    dir_translated =  os.path.join(work_dir,'TRANSLATED_DIRECTORY')
    start_point, stop_point = translate_squad(batch_size=batch_size, use_old_name=use_old_name,
                                              name_to_save=name_to_save_after_translate, args_squad=args_squad,
                                              args_translated=dir_translated, args_cred=cred_file)

    args_fix_translated_dir =  os.path.join(work_dir,'FIX_TRANSLATED_DIRECTORY')
    args_before_fix_translated_file = os.path.join(dir_translated, name_to_save_after_translate)
    fix_translate(args_fix_translated_dir=args_fix_translated_dir,
                  args_before_fix_translated_file=args_before_fix_translated_file,
                  name_to_save=name_to_save_after_translate)

    args_translated_path = os.path.join(args_fix_translated_dir, name_to_save_after_translate)
    get_statistics(start_point=start_point, stop_point=stop_point, args_translated_path=args_translated_path)

    ###########################################  batch2  #############################################################
    batch_size = 1 # 1 is just for fast debug , you can write here 5000 for example
    name_to_save_after_translate = 'batch_2_train.json'
    use_old_name = False
    args_squad = os.path.join(work_dir,'TRANSLATED_DIRECTORY','batch_1_train.json')
    dir_translated =  os.path.join(work_dir,'TRANSLATED_DIRECTORY')
    start_point, stop_point = translate_squad(batch_size=batch_size, use_old_name=use_old_name,
                                              name_to_save=name_to_save_after_translate, args_squad=args_squad,
                                              args_translated=dir_translated, args_cred=cred_file)

    args_fix_translated_dir =  os.path.join(work_dir,'FIX_TRANSLATED_DIRECTORY')
    args_before_fix_translated_file = os.path.join(dir_translated, name_to_save_after_translate)
    fix_translate(args_fix_translated_dir=args_fix_translated_dir,
                  args_before_fix_translated_file=args_before_fix_translated_file,
                  name_to_save=name_to_save_after_translate)

    args_translated_path = os.path.join(args_fix_translated_dir, name_to_save_after_translate)
    get_statistics(start_point=start_point, stop_point=stop_point, args_translated_path=args_translated_path)



    ###########################################  batch3  #############################################################
    batch_size = 1 # 1 is just for fast debug , you can write here 100000 for example
    name_to_save_after_translate = 'batch_3_train.json'
    use_old_name = False
    args_squad = os.path.join(work_dir,'TRANSLATED_DIRECTORY','batch_2_train.json')
    dir_translated =  os.path.join(work_dir,'TRANSLATED_DIRECTORY')
    start_point, stop_point = translate_squad(batch_size=batch_size, use_old_name=use_old_name,
                                              name_to_save=name_to_save_after_translate, args_squad=args_squad,
                                              args_translated=dir_translated, args_cred=cred_file)

    args_fix_translated_dir =  os.path.join(work_dir,'FIX_TRANSLATED_DIRECTORY')
    args_before_fix_translated_file = os.path.join(dir_translated, name_to_save_after_translate)
    fix_translate(args_fix_translated_dir=args_fix_translated_dir,
                  args_before_fix_translated_file=args_before_fix_translated_file,
                  name_to_save=name_to_save_after_translate)

    args_translated_path = os.path.join(args_fix_translated_dir, name_to_save_after_translate)
    get_statistics(start_point=start_point, stop_point=stop_point, args_translated_path=args_translated_path)


def fix_complete_files(work_dir,dev_file_after_translation_path = '', train_file_after_translation_path = ''):

    ###########################################  dev  #############################################################

    name_to_save='validation.json'
    dir_translated =  os.path.join(work_dir,'TRANSLATED_DIRECTORY')
    args_fix_translated_dir =  os.path.join(work_dir,'FIX_TRANSLATED_DIRECTORY')
    args_before_fix_translated_file = os.path.join(dir_translated, 'batch_3_dev.json')
    if dev_file_after_translation_path != '':
        args_before_fix_translated_file=dev_file_after_translation_path
    fix_translate(args_fix_translated_dir=args_fix_translated_dir,
                  args_before_fix_translated_file=args_before_fix_translated_file,
                  name_to_save=name_to_save)



    ###########################################  train  #############################################################

    name_to_save='train.json'
    dir_translated =  os.path.join(work_dir,'TRANSLATED_DIRECTORY')
    args_fix_translated_dir =  os.path.join(work_dir,'FIX_TRANSLATED_DIRECTORY')
    args_before_fix_translated_file = os.path.join(dir_translated, 'batch_3_train.json')
    if train_file_after_translation_path != '':
        args_before_fix_translated_file=train_file_after_translation_path
    fix_translate(args_fix_translated_dir=args_fix_translated_dir,
                  args_before_fix_translated_file=args_before_fix_translated_file,
                  name_to_save=name_to_save)

def statistics_complete_files(work_dir):
    #
    ###########################################  dev  #############################################################
    name_of_file='validation.json'
    args_fix_translated_dir =  os.path.join(work_dir,'FIX_TRANSLATED_DIRECTORY')
    path = os.path.join(args_fix_translated_dir, name_of_file)
    get_statistics(args_translated_path=path)

    ###########################################  train  #############################################################

    name_of_file='train.json'
    args_fix_translated_dir =  os.path.join(work_dir,'FIX_TRANSLATED_DIRECTORY')
    path = os.path.join(args_fix_translated_dir, name_of_file)
    get_statistics(args_translated_path=path)


def remove_bad_tranlations(work_dir):


    ###########################################  dev  #############################################################

    name_of_file='validation.json'
    args_fix_translated_dir =  os.path.join(work_dir,'FIX_TRANSLATED_DIRECTORY')
    args_fix_and_clean_translated_dir =  os.path.join(work_dir,'FIX_AND_CLEAN_TRANSLATED_DIRECTORY')
    path = os.path.join(args_fix_translated_dir, name_of_file)
    remove_bad_translations(args_fix_and_clean_translated_dir=args_fix_and_clean_translated_dir,
                  args_before_fix_translated_file=path,
                  name_to_save=name_of_file)

    ###########################################  train  #############################################################

    name_of_file='train.json'
    args_fix_translated_dir =  os.path.join(work_dir,'FIX_TRANSLATED_DIRECTORY')
    args_fix_and_clean_translated_dir =  os.path.join(work_dir,'FIX_AND_CLEAN_TRANSLATED_DIRECTORY')
    path = os.path.join(args_fix_translated_dir, name_of_file)
    remove_bad_translations(args_fix_and_clean_translated_dir=args_fix_and_clean_translated_dir,
                            args_before_fix_translated_file=path,
                            name_to_save=name_of_file)


def adjust_dataset(work_dir):
    ###########################################  dev  #############################################################

    name_of_file='validation.json'
    args_fix_and_clean_translated_dir = os.path.join(work_dir,'FIX_AND_CLEAN_TRANSLATED_DIRECTORY')
    orig_path_dev = os.path.join(args_fix_and_clean_translated_dir, name_of_file)
    args_hugging_face_format_translated_dir =  os.path.join(work_dir,'HUGGING_FACE_FORMAT_TRANSLATED_DIRECTORY')
    adjust_dataset_format(args_dest_dir=args_hugging_face_format_translated_dir,
                            args_orig_path_file=orig_path_dev,
                            name_to_save=name_of_file)

    ###########################################  train  #############################################################

    name_of_file='train.json'
    args_fix_and_clean_translated_dir =  os.path.join(work_dir,'FIX_AND_CLEAN_TRANSLATED_DIRECTORY')
    orig_path_train = os.path.join(args_fix_and_clean_translated_dir, name_of_file)
    args_hugging_face_format_translated_dir =  os.path.join(work_dir,'HUGGING_FACE_FORMAT_TRANSLATED_DIRECTORY')
    adjust_dataset_format(args_dest_dir=args_hugging_face_format_translated_dir,
                            args_orig_path_file=orig_path_train,
                            name_to_save=name_of_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-wd', '--work_dir', help='Location of work_dir', required=False,default='data_files')
    parser.add_argument('-c', '--cred_file', help='Google Translate Credentials', required=False,default=os.path.join('data_files', 'CREDENTIALS_DIRECTORY','focal-column-336519-878aaf988ecb.json'))
    parser.add_argument('-t', '--translate', help='Flag: True for translate with google api from squad or False if you just want to fix dataset ', required=False , default=False)


    parser.add_argument('-sd', '--squad_dev', help='Location of dev SQuAD file to translate', required=False,default=os.path.join('data_files','SQUAD_DIRECTORY','dev-v1.1.json'))
    parser.add_argument('-st', '--squad_train', help='Location of train SQuAD file to translate', required=False,default=os.path.join('data_files','SQUAD_DIRECTORY','train-v1.1.json'))

    parser.add_argument('-td', '--translated_dev', help='Location of translated dev file for fix', required=False,default='')
    parser.add_argument('-tt', '--translated_train', help='Location of translated train file for fix', required=False,default='')

    args = parser.parse_args()
    work_dir = args.work_dir
    cred_file=args.cred_file
    flag_for_translate=args.translate
    if flag_for_translate:
        dev_squad_file_path = args.squad_dev
        train_squad_file_path = args.squad_train
        dev_file_after_translation_path=''
        train_file_after_translation_path=''
    else:
        dev_file_after_translation_path=args.translated_dev
        train_file_after_translation_path=args.translated_train


    if flag_for_translate:
        translate_dev_and_fix_per_batch(work_dir=work_dir,args_squad=dev_squad_file_path,cred_file=cred_file)
        translate_train_and_fix_per_batch(work_dir=work_dir,args_squad=train_squad_file_path,cred_file=cred_file)

    fix_complete_files(work_dir, dev_file_after_translation_path=dev_file_after_translation_path,
                       train_file_after_translation_path=train_file_after_translation_path)
    statistics_complete_files(work_dir=work_dir)
    remove_bad_tranlations(work_dir=work_dir)
    adjust_dataset(work_dir=work_dir)


    print('Finish all')
