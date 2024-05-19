# LatinWSD
Code for the paper "Language Pivoting from Parallel Corpora for Word Sense Disambiguation of Historical Languages: a Case Study on Latin" presented at LREC-COLING 2024.

## Basic Usage
### Downloading Model and tokenizer
In order to use the code, first download the relevant Latin BERT model and tokenizer from [the relevant repository](https://github.com/dbamman/latin-bert/tree/master/models/subword_tokenizer_latin). For the Latin BERT model, follow the instruction in the [repository](https://github.com/dbamman/latin-bert) to download it. Move both the latin.subword.encoder and the latin_bert folder into base_models.

### Installing libraries
Install the required libraries into your environment with:
```
pip install -r requirements.txt
```

### Run the main script
To fine-tune LatinBERT models on all the target lemmas from semeval dataset with the addition of the Pers_inter data run:
```
python scripts/latin_wsd_bert.py train --bertPath base_models/latin_bert --tokenizerPath base_models/subword_tokenizer_latin/latin.subword.encoder -f data/semeval_wsd_bert.model --max_epochs 20 -i data/semeval_wsd.data -add data/silver_inter_wsd.data -name semeval_with_inter -save -pre -nod
```
Similarly, substitute "silver_inter_wsd.data" with "silver_align_wsd.data" or "silver_rare_wsd.data" to run the models with the addition of Pers_align or Pers_rare datasets, also changing the "-name" option to a different name under which to store the results.

In order to train just on one of the above datasets, instead, run:
```
python scripts/latin_wsd_bert.py train --bertPath base_models/latin_bert --tokenizerPath base_models/subword_tokenizer_latin/latin.subword.encoder -f data/semeval_wsd_bert.model --max_epochs 20 -i data/semeval_wsd.data -name semeval_only -save -pre -nod
```
Again, change "data/semeval_wsd.data" to another dataset among the ones available in the data folder to train on a different dataset.

In all cases, results will be stored in {name}.json file, where {name} is the input to the "-name" option. The models checkpoint will be saved under saved_models in the format {lemma}.bin, where the {lemma} is the target lemma on which that specific instance was fine-tuned on. Make sure to save the checkpoints separately before re-running the experiments with different settings, as they will otherwise be overwritten.
