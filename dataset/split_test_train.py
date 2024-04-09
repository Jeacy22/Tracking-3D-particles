import os
from utils1 import get_config
from pathlib import Path
import numpy as np
train_percent=1
val_percent = 0
if (train_percent+val_percent)>1:
    raise ValueError("the sum of percent is over one")


current_path = Path.cwd()
updated_path = current_path.parent
config = get_config(updated_path)
#annotation_path = config["annotation_path"]
annotation_path = "/root/autodl-tmp/data/test/Annotation"

val_index=[]
train_index=[]

for dirpath,subdir,files in os.walk(annotation_path):
    length = len(files)
    amount_val = val_percent * length
    amount_train = train_percent * length
    indexs = np.random.randint(length,size=int(amount_val))
    for index,name in enumerate(files):
        if index>-1:
            pname = name.split(".")
            if index in indexs:
                val_index.append(pname[0])
            else:
                train_index.append(pname[0])
        else:
            break

split_index_path = config["split_index_path"]
#val_txt = os.path.join(split_index_path,"val.txt")
train_txt = os.path.join(split_index_path,"test.txt")

"""
with open(val_txt,'w+') as f:
    f.write('\n'.join(val_index))
"""

with open(train_txt,'w+') as f:
    f.write('\n'.join(train_index))









