import csv
import inspect
import torch
from torch.utils.data import Dataset
import transformers_embedder as tre
import os
from utils import read_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_labels(languages: list) -> (list, list):
	"""
	function that returns all the possible labels for this task (based on the given dataset)
	:param languages: List of the supported languages
	:return: two ordered list containing the labels for Predicates and Roles
	"""

	SEMANTIC_ROLES = set()
	PREDICATES = set()

	for language in languages:
		_, dy = read_dataset(os.path.join(".", "data", language, "dev.json"))
		_, ty = read_dataset(os.path.join(".", "data", language, "train.json"))
		for k, v in dy.items():
			aux = set().union(*[set(value) for index, value in v["roles"].items()])
			SEMANTIC_ROLES |= aux
		for k, v in ty.items():
			aux = set().union(*[set(value) for index, value in v["roles"].items()])
			SEMANTIC_ROLES |= aux

		for k, v in dy.items():
			PREDICATES |= set(v["predicates"])
		for k, v in ty.items():
			PREDICATES |= set(v["predicates"])

	tsv_file = open(os.path.join(".", "model", "VA_frame_info.tsv"))
	data = csv.reader(tsv_file, delimiter='\t')
	for row in data:
		try:
			PREDICATES.add(row[1].upper())
		except:
			continue
	tsv_file.close()
	PREDICATES.add("_")
	return sorted(PREDICATES), sorted(SEMANTIC_ROLES)


class SharedVars:
	language_model_name12 = 'bert-base-cased'
	language_model_name34 = 'xlm-roberta-base'

	PREDICATES, SEMANTIC_ROLES = get_labels(["EN", "ES", "FR"])
	PREDICATES.append("<PAD>")
	SEMANTIC_ROLES.append("<PAD>")
	semanticRoles2Index = {value: index for index, value in enumerate(SEMANTIC_ROLES)}
	index2semanticRoles = {index: value for index, value in enumerate(SEMANTIC_ROLES)}
	pred2Index = {value: index for index, value in enumerate(PREDICATES)}
	index2pred = {index: value for index, value in enumerate(PREDICATES)}
	tokenizer12 = tre.Tokenizer(language_model_name12)
	tokenizer34 = tokenizer12 if language_model_name12 == language_model_name34 else tre.Tokenizer(
		language_model_name34)


#################################################################################
#                                                                               #
#   The upper part is for sharing values between each dataset and parameters    #
#      class, is placed in this part of the code to be executed only once       #
#                                                                               #
##################################################################################

class HW2Params34:
	"""
	create Class for parameters implementation for model 34
	"""

	lstm_dropout = 0.2
	dropout = 0.1
	bidir = True
	lstm_layers = 1
	lstm_hidden_dim = 128

	hidden = 128

	language_model_name = SharedVars.language_model_name34
	fine_tune = True
	use_pos = False
	batch_size = 16
	gpus = 1
	num_workers = min(4 * gpus, os.cpu_count())
	# optim
	learning_rate = 1e-5
	weight_decay = 0
	pos_vocab_size = 13
	pos_embedding_dim = 20
	# training
	epochs = 15
	n_classes34 = len(SharedVars.SEMANTIC_ROLES)

	def __init__(self, device="cpu"):
		self.device = device

	def gethyperparameterdict(self):
		ret = {}
		for i in inspect.getmembers(self):
			if not i[0].startswith('_'):
				if not inspect.ismethod(i[1]):
					ret[i[0]] = i[1]
		return ret


class HW2Params12:
	"""
	create Class for parameters implementation for model 12
	"""
	lstm_dropout = 0.2
	dropout = 0.1
	bidir = True

	lstm_layers = 1
	lstm_hidden_dim = 512

	hidden = 512

	language_model_name = SharedVars.language_model_name12
	fine_tune = True
	use_pos = False
	batch_size = 16
	gpus = 1
	num_workers = min(4 * gpus, os.cpu_count())
	# optim
	learning_rate = 2e-5
	weight_decay = 0

	# training
	epochs = 30

	def __init__(self, device="cpu"):
		self.device = device

		self.n_classes12 = len(SharedVars.PREDICATES)

	def gethyperparameterdict(self):
		ret = {}
		for i in inspect.getmembers(self):
			if not i[0].startswith('_'):
				if not inspect.ismethod(i[1]):
					ret[i[0]] = i[1]
		return ret


class HW2Dataset34(Dataset):
	"""
	Dataset Class for Homework2 task C and D
	:param sentences: Dict of sentences where each item is a tuple (sentence_id, dict with sentence infos)
	:param labels: labels for the specific task
	:param hassentenceid: boolean value that indicates if the sentence has a sentence_id or not
	"""

	def __init__(self, sentences: dict, labels: dict = None, hassentenceid: bool = True):
		if hassentenceid:
			self.sentences = sentences
		else:
			self.sentences = {"placeholder": sentences}

		self.labels = labels
		self.samples = []
		self._init_data()

	def _init_data(self):
		for s_id, sentence in self.sentences.items():
			for index, word in enumerate(sentence["predicates"]):
				if word != "_":
					pred_sentence = ["_" for _ in range(len(sentence["predicates"]))]
					pred_sentence[index] = word
					sentence_id = s_id + "_" + str(index)

					words_sentence = sentence["words"]
					lemmas_sentence = sentence["lemmas"]
					preds = [0 if (x == "_" or 0) else 1 for x in pred_sentence]
					roles = [SharedVars.semanticRoles2Index[x] for x in
					         self.labels[s_id]["roles"][index]] if self.labels is not None else []

					self.samples.append({
						"sentence_id": sentence_id, "words": words_sentence, "lemmas": lemmas_sentence,
						"predicates": preds, "p_index": index, "roles": roles})

	def __len__(self) -> int:
		"""
		Function that returns the number of samples of the dataset
		:return: returns len of encoded data as integer
		"""
		return len(self.samples)

	def __getitem__(self, idx):
		"""
		Function to get a specific element of the dataset
		:param idx: index of the element
		:return: the element of the corresponding input
		"""
		return self.samples[idx]

	@staticmethod
	def pos2index(treebank_tag):
		"""
		Return wordnet tagset based on the given tag
		:return:
		"""
		if treebank_tag.startswith('ADJ'):
			return 1
		elif treebank_tag.startswith('VERB'):
			return 2
		elif treebank_tag.startswith('PROPN'):
			return 3
		elif treebank_tag.startswith("ADP"):
			return 4
		elif treebank_tag.startswith('ADV'):
			return 5
		elif treebank_tag.startswith('SCONJ'):
			return 6
		elif treebank_tag.startswith('DET'):
			return 7
		elif treebank_tag.startswith('CCONJ'):
			return 8
		elif treebank_tag.startswith('PART'):
			return 9
		elif treebank_tag.startswith('AUX'):
			return 10
		elif treebank_tag.startswith('PUNCT'):
			return 11
		elif treebank_tag.startswith('<PAD>'):
			return 12
		else:
			# NOUN
			return 0

	@staticmethod
	def collate_fn(data: list):
		"""
		Function that is performed on each batch
		:return: a tuple formed by 2 element, one with the information of X and the other of Y
		"""

		sentence_ids, words, dependency_heads, dependency_relations, predicates, ys, indexs = [], [], [], [], [], [], []
		# Unpack the encoded data

		for e in data:
			indexs.append(e["p_index"])
			sentence_ids.append(e["sentence_id"])
			words.append(e["words"])
			predicates.append(e["predicates"])
			labels = ([SharedVars.semanticRoles2Index["<PAD>"]] if SharedVars.tokenizer34.has_starting_token else []) + \
			         e[
				         "roles"] + [
				         SharedVars.semanticRoles2Index["<PAD>"]]
			ys.append(torch.tensor(labels))

		# Pad each input list to the max_len of each batch
		indexs = torch.tensor(indexs)
		ys = torch.nn.utils.rnn.pad_sequence(ys, padding_value=SharedVars.semanticRoles2Index["<PAD>"],
		                                     batch_first=True)
		words = SharedVars.tokenizer34(words, return_tensors=True, padding=True, is_split_into_words=True)
		return {"words": words, "p_indexs": indexs, "bool_cls": SharedVars.tokenizer34.has_starting_token}, {
			"sentence_ids": sentence_ids, "predicates": predicates, "labels": ys}

	@staticmethod
	def decode_labels(enconded_labels: list):
		return [SharedVars.index2semanticRoles[x] for x in enconded_labels]


class HW2Dataset12(Dataset):

	def __init__(self, sentences: dict, labels: dict = None, hassentenceid: bool = True):
		"""
		Dataset Class for Homework2 task C and D
		:param sentences: Dict of sentences where each item is a tuple (sentence_id, dict with sentence infos)
		:param labels: labels for the specific task
		:param hassentenceid: boolean value that indicates if the sentence has a sentence_id or not
		"""
		if hassentenceid:
			self.sentences = sentences
		else:
			self.sentences = {"placeholder": sentences}

		self.labels = labels

		self.samples = []
		self._init_data()

	def _init_data(self):
		for s_id, sentence in self.sentences.items():
			words_sentence = sentence["words"]
			lemmas_sentence = sentence["lemmas"]
			preds = [SharedVars.pred2Index[x] for x in
			         self.labels[s_id]["predicates"]] if self.labels is not None else []
			preds12 = [0 if x == "_" else 1 for x in self.labels[s_id]["predicates"]] if self.labels is not None else []
			self.samples.append(
				{"sentence_id": s_id, "words": words_sentence, "lemmas": lemmas_sentence, "predicates": preds,
				 "predicates12": preds12})

	def __len__(self) -> int:
		"""
		Function that returns the number of samples of the dataset
		:return: returns len of encoded data as integer
		"""
		return len(self.samples)

	def __getitem__(self, idx):
		"""
		Function to get a specific element of the dataset
		:param idx: index of the element
		:return: the element of the corresponding input
		"""
		return self.samples[idx]

	@staticmethod
	def pos2index(treebank_tag):
		"""
		Return wordnet tagset based on the given tag
		:return:
		"""
		if treebank_tag.startswith('ADJ'):
			return 1
		elif treebank_tag.startswith('VERB'):
			return 2
		elif treebank_tag.startswith('PROPN'):
			return 3
		elif treebank_tag.startswith("ADP"):
			return 4
		elif treebank_tag.startswith('ADV'):
			return 5
		elif treebank_tag.startswith('SCONJ'):
			return 6
		elif treebank_tag.startswith('DET'):
			return 7
		elif treebank_tag.startswith('CCONJ'):
			return 8
		elif treebank_tag.startswith('PART'):
			return 9
		elif treebank_tag.startswith('AUX'):
			return 10
		elif treebank_tag.startswith('PUNCT'):
			return 11
		elif treebank_tag.startswith('<PAD>'):
			return 12
		else:
			# NOUN
			return 0

	@staticmethod
	def collate_fn(data: list):
		"""
		Function that is performed on each batch
		:return: a tuple formed by 2 element, one with the information of X and the other of Y
		"""

		sentence_ids, words, dependency_heads, dependency_relations, ys, ys1 = [], [], [], [], [], []
		# Unpack the encoded data

		for e in data:
			sentence_ids.append(e["sentence_id"])
			words.append(e["words"])
			labels = ([SharedVars.pred2Index["<PAD>"]] if SharedVars.tokenizer12.has_starting_token else []) + e[
				"predicates"] + [
				         SharedVars.pred2Index["<PAD>"]]
			labels1 = ([2] if SharedVars.tokenizer12.has_starting_token else []) + e["predicates12"] + [2]
			ys.append(torch.tensor(labels))
			ys1.append(torch.tensor(labels1))
		# Pad each input list to the max_len of each batch
		ys = torch.nn.utils.rnn.pad_sequence(ys, padding_value=SharedVars.pred2Index["<PAD>"], batch_first=True)
		ys1 = torch.nn.utils.rnn.pad_sequence(ys1, padding_value=2, batch_first=True)

		words = SharedVars.tokenizer12(words, return_tensors=True, padding=True, is_split_into_words=True)
		return {"words": words, "bool_cls": SharedVars.tokenizer12.has_starting_token}, {"sentence_ids": sentence_ids,
		                                                                               "labels": ys,
		                                                                               "labels1": ys1}

	@staticmethod
	def decode_labels(enconded_labels: list):
		return [SharedVars.index2pred[x] for x in enconded_labels]
