import json
import random
import os

from typing import Dict, Optional
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .utility import SharedVars, HW2Dataset34, HW2Params34, HW2Dataset12, HW2Params12
from model import Model
import transformers_embedder as tre


def build_model_34(language: str, device: str) -> Model:
	"""
	The implementation of this function is MANDATORY.
	Args:
			language: the model MUST be loaded for the given language
			device: the model MUST be loaded on the indicated device (e.g. "cpu")
	Returns:
			A Model instance that implements steps 3 and 4 of the SRL pipeline.
					3: Argument identification.
					4: Argument classification.
	"""
	model_params = HW2Params34(device=device)
	model_p = os.path.join(".", "model/XLM-34.bkp")
	model = StudentModel34(language=language, params=model_params, eval_type="34")
	model.load_state_dict(torch.load(model_p, map_location=device))
	model.to(device)
	return model


def build_model_234(language: str, device: str) -> Model:
	"""
	The implementation of this function is OPTIONAL.
	Args:
			language: the model MUST be loaded for the given language
			device: the model MUST be loaded on the indicated device (e.g. "cpu")
	Returns:
			A Model instance that implements steps 2, 3 and 4 of the SRL pipeline.
					2: Predicate disambiguation.
					3: Argument identification.
					4: Argument classification.
	"""
	raise NotImplementedError


def build_model_1234(language: str, device: str) -> Model:
	"""
	The implementation of this function is OPTIONAL.
	Args:
			language: the model MUST be loaded for the given language
			device: the model MUST be loaded on the indicated device (e.g. "cpu")
	Returns:
			A Model instance that implements steps 1, 2, 3 and 4 of the SRL pipeline.
					1: Predicate identification.
					2: Predicate disambiguation.
					3: Argument identification.
					4: Argument classification.
	"""

	# load the specific model for the input language
	model12_p = os.path.join(".", "model/BERT-base-12.bkp")
	model12 = StudentModel12(language=language, params=HW2Params12(device=device), eval_type="12")
	model12.load_state_dict(torch.load(model12_p, map_location=device))
	model12.to(device)

	model34_p = os.path.join(".", "model/XLM-34.bkp")
	model34 = StudentModel34(language=language, params=HW2Params34(device=device), eval_type="34")
	model34.load_state_dict(torch.load(model34_p, map_location=device))
	model34.to(device)

	model = StudentModel1234(language="EN", model12=model12, model34=model34)
	model.to(device)

	return model


class Baseline(Model):
	"""
	A very simple baseline to test that the evaluation script works.
	"""

	def __init__(self, language: str, return_predicates=False):
		self.language = language
		self.baselines = Baseline._load_baselines()
		self.return_predicates = return_predicates

	def predict(self, sentence):
		predicate_identification = []
		for pos in sentence["pos_tags"]:
			prob = self.baselines["predicate_identification"].get(pos, dict()).get("positive", 0) / self.baselines[
				"predicate_identification"].get(pos, dict()).get("total", 1)
			if random.random() < prob:
				predicate_identification.append(True)
			else:
				predicate_identification.append(False)

		predicate_disambiguation = []
		predicate_indices = []
		for idx, (lemma, is_predicate) in enumerate(zip(sentence["lemmas"], predicate_identification)):
			if (not is_predicate or lemma not in self.baselines["predicate_disambiguation"]):
				predicate_disambiguation.append("_")
			else:
				predicate_disambiguation.append(self.baselines["predicate_disambiguation"][lemma])
				predicate_indices.append(idx)

		argument_identification = []
		for dependency_relation in sentence["dependency_relations"]:
			prob = self.baselines["argument_identification"].get(dependency_relation, dict()).get("positive", 0) / \
			       self.baselines["argument_identification"].get(dependency_relation, dict()).get("total", 1)
			if random.random() < prob:
				argument_identification.append(True)
			else:
				argument_identification.append(False)

		argument_classification = []
		for dependency_relation, is_argument in zip(sentence["dependency_relations"], argument_identification):
			if not is_argument:
				argument_classification.append("_")
			else:
				argument_classification.append(self.baselines["argument_classification"][dependency_relation])

		if self.return_predicates:
			return {
				"predicates": predicate_disambiguation,
				"roles": {i: argument_classification for i in predicate_indices}, }
		else:
			return {"roles": {i: argument_classification for i in predicate_indices}}

	@staticmethod
	def _load_baselines(path="data/baselines.json"):
		with open(path) as baselines_file:
			baselines = json.load(baselines_file)
		return baselines


class StudentModel12(Model, pl.LightningModule):

	def __init__(self, language: str, params: HW2Params12, eval_type: str):
		super().__init__()
		# load the specific model for the input language
		self.language = language
		self.params = params

		# EMBEDDING LAYERS
		self.word_embedder = tre.TransformersEmbedder(params.language_model_name, subword_pooling_strategy="scatter",
		                                              layer_pooling_strategy="mean", fine_tune=self.params.fine_tune,
		                                              from_pretrained=False)

		combined_len = self.word_embedder.hidden_size

		# Bi-LSTM LAYER
		self.lstm = nn.LSTM(input_size=combined_len, hidden_size=self.params.lstm_hidden_dim,
		                    num_layers=self.params.lstm_layers, bidirectional=self.params.bidir,
		                    dropout=self.params.lstm_dropout if self.params.lstm_layers > 1 else 0, batch_first=True)

		# SHARED LINEARS
		linears = [("lin1",
		            torch.nn.Linear(self.params.lstm_hidden_dim * (2 if self.params.bidir else 1), self.params.hidden)),
		           ("droput", torch.nn.Dropout(self.params.dropout)), ("activation", torch.nn.ReLU()), ]

		# LINEARS FOR TASK A
		linears1 = [("lin1", torch.nn.Linear(self.params.hidden, 3)), ]

		# LINEARS FOR TASK B
		linears2 = [("lin1", torch.nn.Linear(self.params.hidden, self.params.n_classes12)), ]

		# SEQUENTIALS LAYERS FOR CLEANER FORWARD
		self.dual = nn.Sequential(OrderedDict(linears))
		self.classificator1 = nn.Sequential(OrderedDict(linears1))

		self.classificator2 = nn.Sequential(OrderedDict(linears2))

		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.params.learning_rate,
		                                  weight_decay=self.params.weight_decay)

		# LOSS FOR TASK B
		self.loss_fn_c = torch.nn.CrossEntropyLoss(ignore_index=SharedVars.pred2Index["<PAD>"])
		# LOSS FOR TASK A
		self.loss_fn_i = torch.nn.CrossEntropyLoss(ignore_index=2)

	def forward(self, x: dict, y: Optional[dict] = None) -> Dict[str, torch.Tensor]:
		inputs = x["words"]
		out = self.word_embedder(**inputs)
		word_embedding = out.word_embeddings
		lstm_out, _ = self.lstm(word_embedding)
		out = self.dual(lstm_out)
		out1 = self.classificator1(out)
		out2 = self.classificator2(out)
		out1 = out1.permute(0, 2, 1)
		logits1 = torch.softmax(out1, dim=1)
		out2 = out2.permute(0, 2, 1)
		logits2 = torch.softmax(out2, dim=1)

		result = {'logits2': logits2, 'pred2': torch.argmax(logits2, dim=1), 'logits1': logits1,
		          'pred1': torch.argmax(logits1, dim=1)}

		# compute loss
		if y is not None:
			labels = y["labels"]
			labels1 = y["labels1"]
			# while mathematically the CrossEntropyLoss takes as input the probability distributions,
			# torch optimizes its computation internally and takes as input the logits instead
			loss = self.loss(out2, labels, out1, labels1)
			result['loss'] = loss

		return result

	def loss(self, pred, y, predi, yi):
		loss_1 = self.loss_fn_i(predi, yi)
		loss_2 = self.loss_fn_c(pred, y)
		return (loss_1 + (loss_2 * 9)) / 10

	def configure_optimizers(self):
		return self.optimizer

	def predict(self, sentence):
		"""
		--> !!! STUDENT: implement here your predict function !!! <--

		Args:
				sentence: a dictionary that represents an input sentence, for example:
						- If you are doing argument identification + argument classification:
								{
										"words":
												[  "In",  "any",  "event",  ",",  "Mr.",  "Englund",  "and",  "many",  "others",  "say",  "that",  "the",  "easy",  "gains",  "in",  "narrowing",  "the",  "trade",  "gap",  "have",  "already",  "been",  "made",  "."  ]
										"lemmas":
												["in", "any", "event", ",", "mr.", "englund", "and", "many", "others", "say", "that", "the", "easy", "gain", "in", "narrow", "the", "trade", "gap", "have", "already", "be", "make",  "."],
										"predicates":
												["_", "_", "_", "_", "_", "_", "_", "_", "_", "AFFIRM", "_", "_", "_", "_", "_", "REDUCE_DIMINISH", "_", "_", "_", "_", "_", "_", "MOUNT_ASSEMBLE_PRODUCE", "_" ],
								},
						- If you are doing predicate disambiguation + argument identification + argument classification:
								{
										"words": [...], # SAME AS BEFORE
										"lemmas": [...], # SAME AS BEFORE
										"predicates":
												[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
								},
						- If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
								{
										"words": [...], # SAME AS BEFORE
										"lemmas": [...], # SAME AS BEFORE
										# NOTE: you do NOT have a "predicates" field here.
								},

		Returns:
				A dictionary with your predictions:
						- If you are doing argument identification + argument classification:
								{
										"roles": list of lists, # A list of roles for each predicate in the sentence.
								}
						- If you are doing predicate disambiguation + argument identification + argument classification:
								{
										"predicates": list, # A list with your predicted predicate senses, one for each token in the input sentence.
										"roles": dictionary of lists, # A list of roles for each pre-identified predicate (index) in the sentence.
								}
						- If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
								{
										"predicates": list, # A list of predicate senses, one for each token in the sentence, null ("_") included.
										"roles": dictionary of lists, # A list of roles for each predicate (index) you identify in the sentence.
								}
		"""
		sentence_dataset = HW2Dataset12(sentence, hassentenceid=False)
		sentence_dataloader = DataLoader(sentence_dataset, batch_size=self.params.batch_size, shuffle=False,
		                                 collate_fn=HW2Dataset12.collate_fn, num_workers=os.cpu_count())
		pred2 = dict()
		self.eval()
		with torch.no_grad():
			for batch in sentence_dataloader:
				sentences_len = batch[0]["words"]["sentence_lengths"]
				forward_output = self.forward(batch[0])
				for i, v in enumerate(batch[1]["sentence_ids"]):
					cls_index_shift = 1 if batch[0]["bool_cls"] else 0
					# also remove pad
					preds2 = HW2Dataset12.decode_labels(
						(forward_output["pred2"][i][cls_index_shift:sentences_len[i] - (2 - cls_index_shift)]).tolist())
					pred2[v] = {"predicates": preds2}

		return pred2


class StudentModel34(Model, pl.LightningModule):

	def __init__(self, language: str, params: HW2Params34, eval_type="34"):
		super().__init__()
		# load the specific model for the input language
		self.language = language
		self.params = params

		# EMBEDDING LAYERS
		self.word_embedder = tre.TransformersEmbedder(params.language_model_name, subword_pooling_strategy="scatter",
		                                              layer_pooling_strategy="mean", fine_tune=self.params.fine_tune,
		                                              from_pretrained=False)

		# *2 is for the concatenation of the predicate embedding with each token
		combined_len = (self.word_embedder.hidden_size * 2)

		# Bi-LSTM LAYER
		self.lstm = nn.LSTM(input_size=combined_len, hidden_size=self.params.lstm_hidden_dim,
		                    num_layers=self.params.lstm_layers, bidirectional=self.params.bidir,
		                    dropout=self.params.lstm_dropout if self.params.lstm_layers > 1 else 0, batch_first=True)

		# LINEARS FOR TASK C & D
		linears = [("lin1",
		            torch.nn.Linear(self.params.lstm_hidden_dim * (2 if self.params.bidir else 1), self.params.hidden)),
		           ("activation", nn.ReLU()), ("lin2", torch.nn.Linear(self.params.hidden, self.params.n_classes34)), ]

		self.classificator = nn.Sequential(OrderedDict(linears))
		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.params.learning_rate,
		                                  weight_decay=self.params.weight_decay)

		# LOSS FOR TASK C & D
		self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=SharedVars.semanticRoles2Index["<PAD>"])

	def forward(self, x: dict, y: Optional[dict] = None) -> Dict[str, torch.Tensor]:
		inputs = x["words"]
		preds = x["p_indexs"]

		out = self.word_embedder(**inputs)

		word_embedding = out.word_embeddings

		pred_embedding = torch.unsqueeze(word_embedding[range(len(preds)), preds, :], dim=1)
		pred_embedding = pred_embedding.expand(-1, word_embedding.shape[1], -1)
		combined_embeddings = torch.cat((word_embedding, pred_embedding), 2)

		lstm_out, _ = self.lstm(combined_embeddings)

		out = self.classificator(lstm_out)

		out = out.permute(0, 2, 1)
		logits = torch.softmax(out, dim=1)

		result = {'logits': logits, 'pred': torch.argmax(logits, dim=1)}

		# compute loss
		if y is not None:
			labels = y["labels"]
			loss = self.loss(out, labels)
			result['loss'] = loss

		return result

	def loss(self, pred, y):
		return self.loss_fn(pred, y)

	def configure_optimizers(self):
		return self.optimizer

	def predict(self, sentence):
		"""
		--> !!! STUDENT: implement here your predict function !!! <--

		Args:
				sentence: a dictionary that represents an input sentence, for example:
						- If you are doing argument identification + argument classification:
								{
										"words":
												[  "In",  "any",  "event",  ",",  "Mr.",  "Englund",  "and",  "many",  "others",  "say",  "that",  "the",  "easy",  "gains",  "in",  "narrowing",  "the",  "trade",  "gap",  "have",  "already",  "been",  "made",  "."  ]
										"lemmas":
												["in", "any", "event", ",", "mr.", "englund", "and", "many", "others", "say", "that", "the", "easy", "gain", "in", "narrow", "the", "trade", "gap", "have", "already", "be", "make",  "."],
										"predicates":
												["_", "_", "_", "_", "_", "_", "_", "_", "_", "AFFIRM", "_", "_", "_", "_", "_", "REDUCE_DIMINISH", "_", "_", "_", "_", "_", "_", "MOUNT_ASSEMBLE_PRODUCE", "_" ],
								},
						- If you are doing predicate disambiguation + argument identification + argument classification:
								{
										"words": [...], # SAME AS BEFORE
										"lemmas": [...], # SAME AS BEFORE
										"predicates":
												[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
								},
						- If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
								{
										"words": [...], # SAME AS BEFORE
										"lemmas": [...], # SAME AS BEFORE
										# NOTE: you do NOT have a "predicates" field here.
								},

		Returns:
				A dictionary with your predictions:
						- If you are doing argument identification + argument classification:
								{
										"roles": list of lists, # A list of roles for each predicate in the sentence.
								}
						- If you are doing predicate disambiguation + argument identification + argument classification:
								{
										"predicates": list, # A list with your predicted predicate senses, one for each token in the input sentence.
										"roles": dictionary of lists, # A list of roles for each pre-identified predicate (index) in the sentence.
								}
						- If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
								{
										"predicates": list, # A list of predicate senses, one for each token in the sentence, null ("_") included.
										"roles": dictionary of lists, # A list of roles for each predicate (index) you identify in the sentence.
								}
		"""

		sentence_dataset = HW2Dataset34(sentence, hassentenceid=False)
		sentence_dataloader = DataLoader(sentence_dataset, batch_size=self.params.batch_size, shuffle=False,
		                                 collate_fn=HW2Dataset34.collate_fn, num_workers=os.cpu_count())
		pred = dict()
		self.eval()

		with torch.no_grad():
			for batch in sentence_dataloader:
				sentences_len = batch[0]["words"]["sentence_lengths"]
				forward_output = self.forward(batch[0])

				for i, v in enumerate(batch[1]["sentence_ids"]):
					pred_index = batch[0]["p_indexs"][i].item()
					cls_index_shift = 1 if batch[0]["bool_cls"] else 0
					proles = HW2Dataset34.decode_labels(
						(forward_output["pred"][i][cls_index_shift:sentences_len[i] - (2 - cls_index_shift)]).tolist())
					pred[pred_index] = proles

		return {"roles": pred}


class StudentModel1234(Model, pl.LightningModule):

	def __init__(self, language: str, model12: StudentModel12, model34: StudentModel34):
		super().__init__()
		self.model12 = model12
		self.model34 = model34

	def predict(self, sentence):
		out12 = self.model12.predict(sentence)

		sentence["predicates"] = out12.popitem()[1]["predicates"]
		out34 = self.model34.predict(sentence)

		result = {"predicates": sentence["predicates"], "roles": out34["roles"]}

		return result
