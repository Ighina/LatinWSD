import os

import numpy as np
import pandas as pd

new_df = []

to_roman = {1:"I", 2:"II", 3:"III", 4:"IV", 5:"V", 6:"VI", 7:"VII", 8:"VIII",9:"IX"}

for root, _, files in os.walk("AnnotatedLatinISE"):
    for file in files:
        print(file)
        senses = []
        
        lemma = file.split("_")[2]
        if len(lemma.split())>1:
            lemma = lemma.split()[0]
        elif len(lemma.split("-"))>1:
            lemma = lemma.split("-")[0]
        
        annotations = pd.read_excel(os.path.join(root, file))
        annotations.columns = [c.lower() for c in annotations.columns]
        annotations = annotations[annotations["left context"].notna()]

        lemmas = [lemma for _ in range(len(annotations))]

        right_context_column = [idx for idx, column_name in enumerate(annotations.columns) if column_name.startswith("right")][0]
        comment_column = [idx for idx, column_name in enumerate(annotations.columns) if column_name.startswith("comment")][0]
        # print(right_context_column)
        
        for sense_tuple in annotations.iloc[:, right_context_column+1:comment_column].values:
            #print(sense_tuple)
            senses.append(to_roman[np.argmax([float(sense) if len(str(sense))<4 else float(sense[0]) for sense in sense_tuple])+1])

        annotations["sense"]=senses
        annotations["lemmas"] = lemmas
        if "target word" in annotations.columns:
            new_df.extend(annotations[["lemmas", "sense", "left context", "target word", "right context"]].values.tolist()) 
        else:
            new_df.extend(annotations[["lemmas", "sense","left context", "target", "right context"]].values.tolist())

with open("semeval_wsd.data", "w") as f:
    for line in new_df:
        try:
            f.write("\t".join(line)+"\n")
        except TypeError:
            print(line)
            0/0