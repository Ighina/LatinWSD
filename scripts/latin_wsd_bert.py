import os,sys,argparse
from tqdm import tqdm
from transformers import BertModel, BertPreTrainedModel, AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import numpy as np
import sequence_eval
from tensor2tensor.data_generators import text_encoder
import random
import re
import pandas as pd
import json
from sklearn.metrics import f1_score, precision_score, recall_score

random.seed(1)
torch.manual_seed(0)
np.random.seed(0)

batch_size=8
dropout_rate=0.25
bert_dim=768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('')
print("********************************************")
print("Running on: {}".format(device))
print("********************************************")
print('')


def train_test_split(sentences):
    train, dev, test = [], [], []
    if len(sentences)==1:
        return train, dev, test
    elif len(sentences)==2:
        return [sentences[0]], [], [sentences[1]]
    elif len(sentences)<10:
        return sentences[:-2], [sentences[-2]], [sentences[-1]]
    else:
        
        for idx, sentence in enumerate(sentences):
            if not idx%10:
                test.append(sentence)
            elif not idx%9:
                dev.append(sentence)
            else:
                train.append(sentence)
    return train, dev, test


class LatinTokenizer():
  def __init__(self, encoder):
    self.vocab={}
    self.reverseVocab={}
    self.encoder=encoder

    self.vocab["[PAD]"]=0
    self.vocab["[UNK]"]=1
    self.vocab["[CLS]"]=2
    self.vocab["[SEP]"]=3
    self.vocab["[MASK]"]=4
    

    for key in self.encoder._subtoken_string_to_id:
      self.vocab[key]=self.encoder._subtoken_string_to_id[key]+5
      self.reverseVocab[self.encoder._subtoken_string_to_id[key]+5]=key


  def convert_tokens_to_ids(self, tokens):
    wp_tokens=[]
    for token in tokens:
      if token == "[PAD]":
        wp_tokens.append(0)
      elif token == "[UNK]":
        wp_tokens.append(1)
      elif token == "[CLS]":
        wp_tokens.append(2)
      elif token == "[SEP]":
        wp_tokens.append(3)
      elif token == "[MASK]":
        wp_tokens.append(4)

      else:
        wp_tokens.append(self.vocab[token])
    return wp_tokens

  def tokenize(self, text, truncate=False):
    tokens=text.split(" ")
    wp_tokens=[]
    for token in tokens:
      if token in {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}:
        wp_tokens.append(token)
      else:
        wp_toks=self.encoder.encode(token)

        for wp in wp_toks:
          wp_tokens.append(self.reverseVocab[wp+5])
    return wp_tokens if not truncate else wp_tokens[:512]


class BertForSequenceLabeling(nn.Module):

	def __init__(self, tokenizerPath=None, bertPath=None, freeze_bert=False, num_labels=2, original=True):
		super(BertForSequenceLabeling, self).__init__()

		encoder = text_encoder.SubwordTextEncoder(tokenizerPath)
		if original:
			self.tokenizer = LatinTokenizer(encoder)
			self.bert = BertModel.from_pretrained(bertPath)
		else:
			self.tokenizer = AutoTokenizer.from_pretrained("ClassCat/roberta-base-latin-v2")
			self.bert = AutoModel.from_pretrained("ClassCat/roberta-base-latin-v2")
		self.num_labels = num_labels
		

		self.bert.eval()
		
		if freeze_bert:
			for param in self.bert.parameters():
				param.requires_grad = False

		self.dropout = nn.Dropout(dropout_rate)
		self.classifier = nn.Linear(bert_dim, num_labels)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, transforms=None, labels=None):

		input_ids = input_ids.to(device)
		attention_mask = attention_mask.to(device)
		transforms = transforms.to(device)

		if labels is not None:
			labels = labels.to(device)

		sequence_outputs = self.bert(input_ids, token_type_ids=None, attention_mask=attention_mask).last_hidden_state
		#all_layers=sequence_outputs
		out=torch.matmul(transforms,sequence_outputs)

		logits = self.classifier(out)

		if labels is not None:

			loss_fct = CrossEntropyLoss(ignore_index=-100)
			loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			return loss
		else:
			return logits

	def predict(self, dev_file, tagset, outfile):

		num_labels=len(tagset)

		rev_tagset={tagset[v]:v for v in tagset}
		
		dev_orig_sentences = sequence_reader.prepare_annotations_from_file(dev_file, tagset, labeled=False)
		dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, dev_ordering=model.get_batches(dev_orig_sentences, batch_size)

		model.eval()

		bcount=0

		with torch.no_grad():

			ordered_preds=[]

			all_preds=[]
			all_golds=[]

			for b in range(len(dev_batched_data)):
				size=dev_batched_transforms[b].shape
				
				b_size=size[0]
				b_size_labels=size[1]
				b_size_orig=size[2]

				logits = self.forward(dev_batched_data[b], token_type_ids=None, attention_mask=dev_batched_mask[b], transforms=dev_batched_transforms[b])
				
				logits=logits.view(-1, b_size_labels, num_labels)

				logits=logits.cpu()

				preds=np.argmax(logits, axis=2)

				for row in range(b_size):
					ordered_preds.append([np.array(r) for r in preds[row]])
	
			preds_in_order = [None for i in range(len(dev_orig_sentences))]
			for i, ind in enumerate(dev_ordering):
				preds_in_order[ind] = ordered_preds[i]
			
			with open(outfile, "w", encoding="utf-8") as out:
				for idx, sentence in enumerate(dev_orig_sentences):

					# skip [CLS] and [SEP] tokens
					for t_idx in range(1, len(sentence)-1):
						sent_list=sentence[t_idx]
						token=sent_list[0]
						s_idx=sent_list[2]
						filename=sent_list[3]

						pred=preds_in_order[idx][t_idx]

						out.write("%s\t%s\n" % (token, rev_tagset[int(pred)]))

					# longer than just "[CLS] [SEP]"
					if len(sentence) > 2:
						out.write("\n")

	def evaluate(self, dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, metric, num_labels):
		#num_labels=len(tagset)

		model.eval()

		with torch.no_grad():

			ordered_preds=[]

			all_preds=[]
			all_golds=[]
			all_single_pred = []

			for b in range(len(dev_batched_data)):

				logits = self.forward(dev_batched_data[b], token_type_ids=None, attention_mask=dev_batched_mask[b], transforms=dev_batched_transforms[b])

				logits=logits.cpu()

				ordered_preds += [np.array(r) for r in logits]
				size=dev_batched_labels[b].shape

				logits=logits.view(-1, size[1], num_labels)

				for row in range(size[0]):
					for col in range(size[1]):
						if dev_batched_labels[b][row][col] != -100:
							pred=np.argmax(logits[row][col])
							all_preds.append(pred.cpu().numpy())
							all_golds.append(dev_batched_labels[b][row][col].cpu().numpy())
					all_single_pred.append(pred.cpu().numpy())
			
			cor=0.
			tot=0.
			all_cor = []
			for i in range(len(all_golds)):
				if all_golds[i] == all_preds[i]:
					cor+=1
					all_cor.append(1)
				else:
					all_cor.append(0)
				tot+=1

			F1 = f1_score(all_golds, all_preds, average = "weighted")
			precision = precision_score(all_golds, all_preds, average="weighted")
			recall = recall_score(all_golds, all_preds, average="weighted")

			return cor, tot, all_cor, all_single_pred, F1, precision, recall, all_golds, all_preds



	def get_batches(self, sentences, max_batch, preprocess=False):

		maxLen=0
		for sentence in sentences:
			length=0
			for word in sentence:
				if preprocess:
					toks=self.tokenizer.tokenize(preprocess_fn(word[0]), truncate=True)
				else:
					toks=self.tokenizer.tokenize(word[0], truncate=True)
				length+=len(toks)

			if length> maxLen:
				maxLen=min(512, length)

		all_data=[]
		all_masks=[]
		all_labels=[]
		all_transforms=[]

		for sentence in sentences:
			tok_ids=[]
			input_mask=[]
			labels=[]
			transform=[]

			all_toks=[]
			n=0
			word_limit=-1      
			for idx, word in enumerate(sentence):
				toks=self.tokenizer.tokenize(word[0], truncate=True)
				all_toks.append(toks)
				n+=len(toks)
				if n>512 and word_limit<0:
					word_limit=idx   

			cur=0
			for idx, word in enumerate(sentence):
				toks=all_toks[idx]
				ind=list(np.zeros(min(512,n)))
				for j in range(cur,cur+len(toks)):
					if j<len(ind):
					  ind[j]=1./len(toks)
				cur+=len(toks)
				transform.append(ind)

				tok_ids.extend(self.tokenizer.convert_tokens_to_ids(toks))

				input_mask.extend(np.ones(len(toks)))
				labels.append(int(word[1]))

			all_data.append(tok_ids[:512])
			all_masks.append(input_mask[:512])
			all_labels.append(labels[:word_limit])
			all_transforms.append(transform[:word_limit])

		lengths = np.array([len(l) for l in all_data])

		# Note sequence must be ordered from shortest to longest so current_batch will work
		ordering = np.argsort(lengths)
		
		ordered_data = [None for i in range(len(all_data))]
		ordered_masks = [None for i in range(len(all_data))]
		ordered_labels = [None for i in range(len(all_data))]
		ordered_transforms = [None for i in range(len(all_data))]
		

		for i, ind in enumerate(ordering):
			ordered_data[i] = all_data[ind]
			ordered_masks[i] = all_masks[ind]
			ordered_labels[i] = all_labels[ind]
			ordered_transforms[i] = all_transforms[ind]

		batched_data=[]
		batched_mask=[]
		batched_labels=[]
		batched_transforms=[]

		i=0
		current_batch=max_batch

		while i < len(ordered_data):

			batch_data=ordered_data[i:i+current_batch]
			batch_mask=ordered_masks[i:i+current_batch]
			batch_labels=ordered_labels[i:i+current_batch]
			batch_transforms=ordered_transforms[i:i+current_batch]

			max_len = max([len(sent) for sent in batch_data])
			max_label = max([len(label) for label in batch_labels])

			for j in range(len(batch_data)):
				
				blen=len(batch_data[j])
				blab=len(batch_labels[j])

				for k in range(blen, max_len):
					batch_data[j].append(0)
					batch_mask[j].append(0)
					for z in range(len(batch_transforms[j])):
						batch_transforms[j][z].append(0)

				for k in range(blab, max_label):
					batch_labels[j].append(-100)

				for k in range(len(batch_transforms[j]), max_label):
					batch_transforms[j].append(np.zeros(max_len))

			batched_data.append(torch.LongTensor(batch_data))
			batched_mask.append(torch.FloatTensor(batch_mask))
			batched_labels.append(torch.LongTensor(batch_labels))
			batched_transforms.append(torch.FloatTensor(batch_transforms))

			bsize=torch.FloatTensor(batch_transforms).shape
			
			i+=current_batch

			# adjust batch size; sentences are ordered from shortest to longest so decrease as they get longer
			if max_len > 100:
				current_batch=12
			if max_len > 200:
				current_batch=6

		return batched_data, batched_mask, batched_labels, batched_transforms, ordering


def get_splits(data):
	trains=[]
	tests=[]
	devs=[]

	for i in range(10):
		trains.append([])
		tests.append([])
		devs.append([])

	for sense_id in range(len(data)):
		train, dev, test = train_test_split(data[sense_id])
		trains[0].extend(train)
		devs[0].extend(dev)
		tests[0].extend(test)
# 		for idx, sent in enumerate(data[sense_id]):
# 			testFold=idx % 10
# 			devFold=testFold-1
# 			if devFold == -1:
# 				devFold=9

# 			for i in range(10):
# 				if i == testFold:
# 					tests[i].append(sent)
# 				elif i == devFold:
# 					devs[i].append(sent)
# 				else:
# 					trains[i].append(sent)
	
	# for idx, sent in enumerate(data[1]):
	# 	testFold=idx % 10
	# 	devFold=testFold-1
	# 	if devFold == -1:
	# 		devFold=9

	# 	for i in range(10):
	# 		if i == testFold:
	# 			tests[i].append(sent)
	# 		elif i == devFold:
	# 			devs[i].append(sent)
	# 		else:
	# 			trains[i].append(sent)
	
	# for idx, sent in enumerate(data[2]):
	# 	testFold=idx % 10
	# 	devFold=testFold-1
	# 	if devFold == -1:
	# 		devFold=9

	# 	for i in range(10):
	# 		if i == testFold:
	# 			tests[i].append(sent)
	# 		elif i == devFold:
	# 			devs[i].append(sent)
	# 		else:
	# 			trains[i].append(sent)
	
	# for idx, sent in enumerate(data[3]):
	# 	testFold=idx % 10
	# 	devFold=testFold-1
	# 	if devFold == -1:
	# 		devFold=9

	# 	for i in range(10):
	# 		if i == testFold:
	# 			tests[i].append(sent)
	# 		elif i == devFold:
	# 			devs[i].append(sent)
	# 		else:
	# 			trains[i].append(sent)
	
	# for idx, sent in enumerate(data[4]):
	# 	testFold=idx % 10
	# 	devFold=testFold-1
	# 	if devFold == -1:
	# 		devFold=9

	# 	for i in range(10):
	# 		if i == testFold:
	# 			tests[i].append(sent)
	# 		elif i == devFold:
	# 			devs[i].append(sent)
	# 		else:
	# 			trains[i].append(sent)

	for i in range(10):
		random.shuffle(trains[i])
		random.shuffle(tests[i])
		random.shuffle(devs[i])

	return trains, devs, tests

def get_labs(before, target, after, label):
	sent=[]
	for word in before.split(" "):
		sent.append((preprocess_fn(word), -100))
	sent.append((preprocess_fn(target), label))
	for word in after.split(" "):
		sent.append((preprocess_fn(word), -100))
	return sent

def preprocess_fn(sentence):
    return re.sub("[^A-Za-z ]", "", sentence)

def read_data(filename, mapping=None):
	if mapping is None:
		lemmas={}
		lemmas_label_map={}
	else:
		lemmas={k:[] for k in mapping.keys()}
		lemmas_label_map=mapping
	try:
		with open(filename, encoding="utf-8") as file:
			for line in file:
				try:
					cols=line.split("\t")
					lemma=cols[0]
					label=cols[1]
					before=cols[2]
					target=cols[3]
					after=cols[4].rstrip()
				except IndexError:
					continue
				if lemma not in lemmas:
					lemmas[lemma]=[]
					if mapping is None:
						lemmas[lemma]={}
						lemmas_label_map[lemma]={}
				if mapping is None and label not in lemmas_label_map[lemma]:
						lemmas_label_map[lemma][label]=len(lemmas[lemma])
						lemmas[lemma][lemmas_label_map[lemma][label]]=[]
				# 	lemmas[lemma][1]=[]
				# 	lemmas[lemma][2]=[]
				# 	lemmas[lemma][3]=[]
				# 	lemmas[lemma][4]=[]
					
				# if label == "I":
				# 	lemmas[lemma][0].append(get_labs(before, target, after, 0))
				# elif label == "II":
				# 	lemmas[lemma][1].append(get_labs(before, target, after, 1))
				# elif label == "III":
				# 	lemmas[lemma][2].append(get_labs(before, target, after, 2))
				# elif label == "IV":
				# 	lemmas[lemma][3].append(get_labs(before, target, after, 3))
				# elif label == "V":
				# 	lemmas[lemma][4].append(get_labs(before, target, after, 4))
				if mapping is None:
					#print(lemmas_label_map)
					lemmas[lemma][lemmas_label_map[lemma][label]].append(get_labs(before, target, after, lemmas_label_map[lemma][label]))
				else:
					try:
						lemmas[lemma].append(get_labs(before, target, after, mapping[lemma][label]))
					except KeyError:
						#print(lemma)
						#print(label)
						continue
	
	except UnicodeDecodeError:
		0/0
		with open(filename, encoding="ISO-8859-1") as file:
			for line in file:
				cols=line.split("\t")
				lemma=cols[0]
				label=cols[1]
				before=cols[2]
				target=cols[3]
				after=cols[4].rstrip()
				if lemma not in lemmas:
					lemmas[lemma]={}
					lemmas[lemma][0]=[]
					lemmas[lemma][1]=[]
					lemmas[lemma][2]=[]
					lemmas[lemma][3]=[]
					lemmas[lemma][4]=[]
					
				if label == "I":
					lemmas[lemma][0].append(get_labs(before, target, after, 0))
				elif label == "II":
					lemmas[lemma][1].append(get_labs(before, target, after, 1))
				elif label == "III":
					lemmas[lemma][2].append(get_labs(before, target, after, 2))
				elif label == "IV":
					lemmas[lemma][3].append(get_labs(before, target, after, 3))
				elif label == "V":
					lemmas[lemma][4].append(get_labs(before, target, after, 4))

	return lemmas, lemmas_label_map

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--mode', help='{train,test,predict,predictBatch}', required=True)
	parser.add_argument('--bertPath', help='path to pre-trained BERT', required=True)
	parser.add_argument('--tokenizerPath', help='path to Latin WordPiece tokenizer', required=True)	
	parser.add_argument('-f','--modelFile', help='File to write model to/read from', required=False)
	parser.add_argument('--max_epochs', help='max number of epochs', required=False)
	parser.add_argument('-i','--inputFile', help='WSD data', required=False)
	parser.add_argument('-save', '--save_models', help='Save the weights of each lemma wsd model', action='store_true')
	parser.add_argument('-jpl', '--just_perfect_lemma', help='Run the training just for the lemmas for which we know we will get perfect accuracy (specified in all_accuracies.csv)', action='store_true')
	parser.add_argument('-skl', '--skip_list', help='Run the training just for the lemmas that we have specified (the ones not in the skip list)', action='store_true')
	parser.add_argument('-inf', '--inference', action = 'store_true', help = "whether to perform inference instead of training routine")
	parser.add_argument('-no', '--no_original', action = 'store_false', help = "use roberta latin rather than the original bert")
	parser.add_argument('-pre', '--preprocess', action='store_true', help="whether to preprocess sentences to leave just alphanumeric characters.")
	parser.add_argument('-add', '--add_silver', required=False, help="The silver dataset to add to the training data. If not included, it will not use it.")
	parser.add_argument('-name', '--experiment_name', default="new_experiment", help="The name of the experiment, which will be included in the output files.")
	parser.add_argument('-nod', '--no_development', action="store_true", help="If true, use last epoch to save model instead of best development epoch.")
	parser.add_argument('-not', '--no_original_train', action="store_true", help="If true, just use the silver dataset to train.")
	args = vars(parser.parse_args())

	print(args)

	mode=args["mode"]
	add_save_folder = "" if args["no_original"] else "_roberta"
	
	inputFile=args["inputFile"]

	modelFile=args["modelFile"]
	max_epochs=args["max_epochs"]

	bertPath=args["bertPath"]
	tokenizerPath=args["tokenizerPath"]

	data,lemmas_label_map=read_data(inputFile, mapping=None)
	if args["add_silver"] is not None:
		extra_data, _ = read_data(args["add_silver"],mapping=lemmas_label_map)

	with open("lemma_label_map_"+os.path.split(inputFile)[1], "w") as f:
		json.dump(lemmas_label_map, f)

	tagset={0:0, 1:1, 2:2, 3:3, 4:4}

	epochs=100

	devCors=[0.]*epochs
	testCors=[0.]*epochs
	devN=[0.]*epochs
	testN=[0.]*epochs
	AllTest = [[0.]*epochs for _ in data]
    
	result_df = {"lemma":[], "accuracy": [], "sentence": [], "sense": [], "epoch":[], "prediction": []}
	result_df_token = {"lemma":[], "sense": [], "epoch":[], "prediction": []}
	metrics = {"F1":[0. for x in range(epochs)], "precision":[0. for x in range(epochs)], "recall":[0. for x in range(epochs)]}
	if args["just_perfect_lemma"]:
		accuracies = pd.read_csv("all_accuracies.csv")
		lemmas_to_skip = set(accuracies[accuracies.accuracy<1]["lemma"].values.tolist())
	if args["skip_list"]:
		lemmas_to_skip = set(['ab', 'adeo2', 'aliquis', 'alius2', 'alter', 'altus1', 'an1', 'ante', 'apud', 'at', 'atque', 'aut', 'contra', 'cum1', 'cur', 'dum', 'et', 'etiam', 'ex', 'hic',  'ibi', 'ille', 'in1', 'ipse', 'jam', 'modo', 'nam', 'nunc', 'omnino', 'quam', 'satis', 'sed1', 'semel', 'si','sic', 'sicut', 'simul', 'sub', 'sui', 'super2', 'supra', 'suus', 'tam', 'tamen', 'tum', 'tunc', 'ubi', 'unde',        'unus', 'usque', 'ut','res', 'habeo', 'sum'])
	else:
		lemmas_to_skip = set([])

	if mode == "train":
	
		metric=sequence_eval.get_accuracy
		denominator = 1
		for lemma_idx, lemma in enumerate(data):
			if lemma in lemmas_to_skip:
				continue

			cor=0.
			tot=0.

			print(lemma)
			num_senses = len(lemmas_label_map[lemma])
			if args["inference"]:
				tests = data[lemma]
			else:
				trains, devs, tests=get_splits(data[lemma])
# 			result_df["lemma"].append(lemma)

				trainData=trains[0]+extra_data[lemma] if args["add_silver"] is not None else trains[0]
				trainData=extra_data[lemma] if args["no_original_train"] else trainData
				devData=devs[0]

			testData=tests[0]+devData if args["no_development"] else tests[0]
			print(len(trainData))
			model = BertForSequenceLabeling(tokenizerPath=tokenizerPath, bertPath=bertPath, freeze_bert=False, num_labels=num_senses, original=args["no_original"])

			model.to(device)

			if not args["inference"]:
				batched_data, batched_mask, batched_labels, batched_transforms, ordering=model.get_batches(trainData, batch_size)
				#model.load_state_dict(torch.load(os.path.join("saved_models", f"{lemma}.bin")))
				dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, dev_ordering=model.get_batches(devData, batch_size)
				assert len(dev_batched_data[0])>0
				
			
			else:
				epochs=1
				model.load_state_dict(torch.load(os.path.join("saved_models"+add_save_folder, f"{lemma}.bin")))
      		
			test_batched_data, test_batched_mask, test_batched_labels, test_batched_transforms, test_ordering=model.get_batches(testData, batch_size)

			learning_rate_fine_tuning=0.00005
			optimizer = optim.Adam(model.parameters(), lr=learning_rate_fine_tuning)
			
			maxScore=0
			best_idx=0

			if max_epochs is not None:
				epochs=int(max_epochs)

			for epoch in range(epochs):
				if not args["inference"]:
					model.train()
					big_loss=0
					for b in range(len(batched_data)):
						if b % 10 == 0:
							# print(b)
							sys.stdout.flush()
						
						loss = model(batched_data[b], token_type_ids=None, attention_mask=batched_mask[b], transforms=batched_transforms[b], labels=batched_labels[b])
						big_loss+=loss
						loss.backward()
						optimizer.step()
						model.zero_grad()

					c, t, all_cor_v,_,_,_,_,_,_=model.evaluate(dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, metric, num_senses)
					assert t>0
					devCors[epoch]+=c
					devN[epoch]+=t

				c, t, all_cor_t,all_preds,F1,precision,recall, gold_tokens, pred_tokens=model.evaluate(test_batched_data, test_batched_mask, test_batched_labels, test_batched_transforms, metric, num_senses)
				#print(len(testData))
				#print(len(all_preds))
				testCors[epoch]+=c
				testN[epoch]+=t
				AllTest[lemma_idx][epoch] = c/t
				metrics["F1"][epoch]+=F1
				metrics["precision"][epoch]+=precision
				metrics["recall"][epoch]+=recall
				metrics["F1"][epoch]/=denominator
				metrics["precision"][epoch]/=denominator
				metrics["recall"][epoch]/=denominator
				denominator = 2				
				cor_idx = 0
				for ind, idx in enumerate(test_ordering):
					sent = ""
					label = -100
					for word in testData[idx]:
						sent+=word[0]+' '
						if word[1]>-100:
							label = word[1]

					result_df["sentence"].append(sent)
					
					if not args["inference"]:
						result_df["sense"].append(label)
						result_df["prediction"].append(all_preds[ind])
						result_df_token["sense"].extend([g.item() for g in gold_tokens]) # TODO: the current evaluation is at the token level which creates discrepancies. Try to create a final dataframe with results at the token level.
						result_df_token["prediction"].extend([p.item() for p in pred_tokens])
						result_df_token["lemma"].extend([epoch for _ in range(len(pred_tokens))])
						result_df_token["epoch"].extend([lemma for _ in range(len(pred_tokens))])
						try:
							all_preds[idx]/1
						except:
							print(all_preds[idx])
							0/0
					else:
						result_df["sense"].append(all_preds[ind])
					result_df["accuracy"].append(all_cor_t[cor_idx])
					result_df["lemma"].append(lemma)
					result_df["epoch"].append(epoch)
					cor_idx+=1
            
			if not args["inference"]:        
				for epoch in range(epochs):
					devAcc=devCors[epoch]/devN[epoch]
					print("DEV:\t%s\t%.3f\t%s\t%s" % (epoch, devAcc, lemma, devN[epoch]))
					sys.stdout.flush()
				
				if args["save_models"]:
					if args["no_original"]:
							PATH=f"saved_models/{lemma}.bin"
					else:
							PATH=f"saved_models_roberta/{lemma}.bin"
					try:
							torch.save(model.state_dict(), PATH)
					except RuntimeError:
							print("could not save the model")

		if not args["inference"]:
			maxAcc=0
			bestDevEpoch=None
			for i in range(epochs):
				acc=devCors[i]/devN[i]
				if acc > maxAcc:
					maxAcc=acc
					bestDevEpoch=epochs-1 if args["no_development"] else i

			testAcc=testCors[bestDevEpoch]/testN[bestDevEpoch]
			final_metrics = {"F1":metrics["F1"][bestDevEpoch], "precision":metrics["precision"][bestDevEpoch], "recall":metrics["recall"][bestDevEpoch], "accuracy":testAcc}
			for metric,value in final_metrics.items():
				print(f"Final {metric}: {value}")
			with open(f"test_scores_{args['experiment_name']}.json", "w") as f:
				json.dump(final_metrics, f)
			#for i in range(len(AllTest)):
			#	result_df["accuracy"].append(AllTest[i][bestDevEpoch])
		else:
			testAcc=0
			bestDevEpoch=0
		print("OVERALL:\t%s\t%.3f\t%s" % (bestDevEpoch, testAcc, testN[bestDevEpoch]))
		result_df["prediction"] = [x.tolist() for x in result_df["prediction"]]
		assert len(result_df["prediction"])==len(result_df["sense"])
		try:
			with open(f"results_{args['experiment_name']}.json", "w") as f:
				json.dump(result_df, f)
			
		except:
			print(result_df["prediction"])
		
		with open(f"results_tokens_{args['experiment_name']}.json", "w") as f:
				json.dump(result_df_token, f)

		


