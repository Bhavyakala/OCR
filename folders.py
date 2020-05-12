import os
import shutil
import split_folders

DIR = 'D:/Coding_wo_cp/OCR/by_class/by_class'

for p in os.listdir(DIR) :
    CUR_DIR = os.path.join(DIR,p)
    for f in os.listdir(CUR_DIR) :
        # print(f)
        if f.startswith("hsf"):
            path = os.path.join(CUR_DIR,f)
            if os.path.isdir(path) :
                # print(path)
                shutil.rmtree(path)
            elif os.path.isfile(path) :
                os.remove(path)    
            # print(f)   
c =0       
for p in os.listdir(DIR) :
    CUR_DIR = os.path.join(DIR,p)
    d = os.listdir(CUR_DIR)
    if len(d)==1:
        for f in os.listdir(os.path.join(CUR_DIR,d[0])) :
            shutil.move(os.path.join(os.path.join(CUR_DIR,d[0]),f), os.path.join(CUR_DIR,f))

split_folders.ratio('D:/Coding_wo_cp/OCR/by_class/by_class',output='data',ratio=(0.8,0.1,0.1))