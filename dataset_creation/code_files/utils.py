from shutil import copyfile
import os
from pathlib import Path
# import easygui as e
import webbrowser

HEBREW_LETTERS = 'אבגדהוזחטיכלמנסעפצקרשת'


def is_hebrew(text):
    for letter in text:
        if HEBREW_LETTERS.find(letter) > 0:
            return True
    return False


def make_copy_of_json(orginal_file , dst_dir,name_of_new_file,use_old_name=False):

    # Check whether the specified path exists or not
    isExist = os.path.exists(dst_dir)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dst_dir)


    if use_old_name is True:
        name_of_file=os.path.basename(orginal_file)
        name_of_file=name_of_new_file+'_'+name_of_file
    else:
        name_of_file=name_of_new_file
    copy_file_dst=os.path.join(dst_dir,name_of_file)

    # my_file = Path(copy_file_dst)
    # if my_file.is_file():
    #     # e.msgbox("File exists! Please Remove it :(", "Error")
    #     webbrowser.open(dst_dir)
    #     raise Exception("File exists! Please Remove it. Dir:",dst_dir)
    # else:
    #     copyfile(orginal_file, copy_file_dst)
    copyfile(orginal_file, copy_file_dst)

    return copy_file_dst