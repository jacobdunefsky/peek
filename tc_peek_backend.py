import gc
import enum
import os
import copy
import json
import dataclasses
import numpy as np
import zipfile

import os.path

if __package__ is not None and __package__ != "":
	from . import render
	from . import backend_import_cfg
else:
	import render
	import backend_import_cfg

from dataclasses import dataclass
from typing import Optional, List, Dict
from threading import RLock

# okay, in this next part, we're gonna do some real funky stuff
# for some background: torch, huggingface_hub, and especially transformer_lens can expensive libraries to load
# if we're just rendering a session, then we don't need to load them
# as such, we only want to load them if necessary
# to do this, we'll create "dummy classes" that pretend to be the modules in question, but do nothing
# then, if the flag backend_import_cfg.import_expensive_modules is set, then we use the real modules
# otherwise, we use the dummies

class Dummy:
	def __init__(self, *args, name=""):
		self.name = name
	def __getattr__(self, other):
		new_obj = type(self)(name=self.name+'.'+other)
		return new_obj

class DecoratorDummy(Dummy):
	def __call__(self, *args, **kwargs):
		def decorator(func):
			def wrapper(*args, **kwargs):
				return func(*args, **kwargs)
			return wrapper
		return decorator

if backend_import_cfg.import_expensive_modules:
	import torch
	import safetensors
	import huggingface_hub as hf_hub
	from transformer_lens import HookedTransformer

	no_grad = torch.no_grad
	if torch.cuda.is_available():
		device = 'cuda'
	else:
		device = 'cpu'
	
	dtype = torch.float32

else:
	torch = Dummy(name='torch')
	hf_hub = Dummy()
	safetensors = Dummy()
	HookedTransformer = Dummy()	

	no_grad	= DecoratorDummy(name='torch.no_grad')

DTYPE_DICT = {
	None: torch.float32,
	'float64': torch.float64,
	'float32': torch.float32,
	'float16': torch.float16,
	'bfloat16': torch.bfloat16,
	'uint8': torch.uint8,
	'int8': torch.int8,
	'int16': torch.int16,
	'int32': torch.int32,
	'int64': torch.int64,

	'torch.float64': torch.float64,
	'torch.float32': torch.float32,
	'torch.float16': torch.float16,
	'torch.bfloat16': torch.bfloat16,
	'torch.uint8': torch.uint8,
	'torch.int8': torch.int8,
	'torch.int16': torch.int16,
	'torch.int32': torch.int32,
	'torch.int64': torch.int64,
}

def _to_numpy(x):
	x = x.detach().cpu()
	try:
		x = x.numpy()
	except TypeError:
		x = x.to(torch.float32).numpy()
	return x

# NOTE: currently, doesn't support sublayers beyond the following
class Sublayer(enum.Enum):
	PRE_LOGITS = enum.auto() # post-ln_f
	LOGITS = enum.auto() # post unembed

	EMBED = enum.auto()
	RESID_PRE = enum.auto()
	ATTN_IN = enum.auto() # post-LN1
	ATTN_OUT = enum.auto() # not necessarily same as resid_mid because of parallel models (curse you Pythia!)
	RESID_MID = enum.auto()
	MLP_IN = enum.auto() # post-LN2
	MLP_OUT = enum.auto()
	RESID_POST = enum.auto() # same as mlp_out

	def __str__(self):
		if self == type(self).EMBED:
			return "embed"
		if self == type(self).RESID_PRE:
			return 'pre'
		elif self == type(self).ATTN_IN:
			return 'attn_in'
		elif self == type(self).ATTN_OUT:
			return 'attn_out'
		elif self == type(self).RESID_MID:
			return 'mid'
		elif self == type(self).MLP_IN:
			return 'mlp_in'
		elif self == type(self).MLP_OUT:
			return 'mlp_out'
		elif self == type(self).RESID_POST:
			return 'post'
		elif self == type(self).PRE_LOGITS:
			return 'pre_logits'
		elif self == type(self).LOGITS:
			return 'logits'
	
	def serialize(self):
		return str(self)

	@classmethod
	def deserialize(cls, s):
		if s == "embed":
			return cls.EMBED
		if s == 'pre':
			return cls.RESID_PRE
		elif s == 'attn_in':
			return cls.ATTN_IN
		elif s == 'attn_out':
			return cls.ATTN_OUT
		elif s == 'mid':
			return cls.RESID_MID
		elif s == 'mlp_in':
			return cls.MLP_IN
		elif s == 'mlp_out':
			return cls.MLP_OUT
		elif s == 'post':
			return cls.RESID_POST
		elif s == 'pre_logits':
			return cls.PRE_LOGITS
		elif s == 'logits':
			return cls.LOGITS

class LayerSublayer:
	def __init__(self, layer, sublayer, parallel_attn_mlp=False):
		self.layer = layer
		self.sublayer = sublayer
		self.parallel_attn_mlp = parallel_attn_mlp

	@classmethod
	def from_hookpoint_str(cls, hookpoint):
		# TODO: hookpoint str validation
		if hookpoint == "hook_embed":
			return cls(0, Sublayer.EMBED)
		split_str = hookpoint.split(".")
		layer = int(split_str[1])
		if split_str[-1] == 'hook_normalized':
			if split_str[2] == 'ln1':
				sublayer = Sublayer.ATTN_IN
			elif split_str[2] == 'ln2':
				sublayer = Sublayer.MLP_IN
			elif split_str[2] == 'ln_f':
				sublayer = Sublayer.PRE_LOGITS
		elif split_str[2] == 'hook_resid_pre':
			sublayer = Sublayer.RESID_PRE
		elif split_str[2] == 'hook_resid_mid':
			sublayer = Sublayer.RESID_MID
		elif split_str[2] == 'hook_attn_out':
			sublayer = Sublayer.ATTN_OUT
		elif split_str[2] == 'hook_mlp_out':
			sublayer = Sublayer.MLP_OUT
		elif split_str[2] == 'hook_resid_post':
			sublayer = Sublayer.RESID_POST
		retval = cls(layer, sublayer)
		return retval

	def to_hookpoint_str(self):
		if self.sublayer == Sublayer.EMBED:
			return "hook_embed"

		if self.sublayer == Sublayer.RESID_PRE:
			suffix = 'hook_resid_pre'
		elif self.sublayer == Sublayer.ATTN_IN:
			suffix = 'ln1.hook_normalized'
		elif self.sublayer == Sublayer.ATTN_OUT:
			suffix = 'hook_attn_out'
		elif self.sublayer == Sublayer.RESID_MID:
			suffix = 'hook_resid_mid'
		elif self.sublayer == Sublayer.MLP_IN:
			suffix = 'ln2.hook_normalized'
		elif self.sublayer == Sublayer.MLP_OUT:
			suffix = 'hook_mlp_out'
		elif self.sublayer == Sublayer.RESID_POST:
			suffix = 'hook_resid_post'
		elif self.sublayer == Sublayer.PRE_LOGITS:
			return 'ln_final.hook_normalized'
		elif self.sublayer == Sublayer.LOGITS:
			return None # logits don't have a hook point

		try:
			return f'blocks.{self.layer}.{suffix}'
		except:
			print(self.layer, self.sublayer)
			raise Exception

	# intended for use in ordinal comparisons -- actual values 
	def val(self):
		sublayer_val = self.sublayer.value
		if self.sublayer == Sublayer.ATTN_OUT:
			if self.parallel_attn_mlp:
				sublayer_val = Sublayer.RESID_POST.value
			else:
				sublayer_val = Sublayer.RESID_MID.value
		elif self.sublayer == Sublayer.MLP_IN:
			if self.parallel_attn_mlp:
				sublayer_val = Sublayer.ATTN_IN.value
		elif self.sublayer == Sublayer.MLP_OUT:
			sublayer_val = Sublayer.RESID_POST.value

		return self.layer + sublayer_val/Sublayer.RESID_POST.value
	def __lt__(self, other):
		return self.val() < other.val()
	def __le__(self, other):
		return self.val() <= other.val()

	def __eq__(self, other):
		return self.layer == other.layer and self.sublayer == other.sublayer
	def __str__(self):
		return f'{self.layer}{str(self.sublayer)}'

	def serialize(self):
		return { 'layer': self.layer, 'sublayer': self.sublayer.serialize(), 'parallel_attn_mlp': self.parallel_attn_mlp }
	@classmethod
	def deserialize(cls, d):
		return cls(
			layer=d['layer'],
			sublayer=Sublayer.deserialize(d['sublayer']),
			parallel_attn_mlp=d['parallel_attn_mlp']
		)

### FeatureInfo-related classes/methods ###
class FeatureType(enum.Enum):
	SAE = 'sae'
	OBSERVABLE = 'observable'
	PULLBACK = 'pullback'
	OTHER = 'other'

	def serialize(self):
		return self.value
	@classmethod
	def deserialize(cls, s):
		return {x.value: x for x in cls}[s]

class FeatureInfo:
	def __init__(self):
		self.name = ''
		self.description = ''

		self.encoder_vector = None
		self.encoder_bias = 0
		self.decoder_vector = None

		self.input_layer : LayerSublayer = None
		self.output_layer : LayerSublayer = None

		self.feature_type : FeatureType = None
		self.use_relu : bool = False

		self.attn_head = None

		self.observable_tokens = None
		self.observable_weights = None

		self.sae_info = None
		self.feature_idx = None
	
	def serialize(self):
		retdict = {
			'name': self.name,
			'description': self.description,
			'encoder_bias': self.encoder_bias,
			'input_layer': self.input_layer.serialize(),
			'output_layer': self.output_layer.serialize(),
			'feature_type': self.feature_type.serialize(),
			'use_relu': self.use_relu,
			'attn_head': self.attn_head,
			'observable_tokens': self.observable_tokens,
			'observable_weights': self.observable_weights,
			'sae_info': self.sae_info.serialize() if self.sae_info is not None else None,
			'feature_idx': self.feature_idx,
			'metadata_only': self.encoder_vector is None or self.decoder_vector is None
		}

		return retdict

	@classmethod
	def deserialize(cls, d):
		new_info = cls()

		new_info.name = d['name']
		new_info.description = d['description']
		new_info.encoder_bias = d['encoder_bias']
		new_info.input_layer = LayerSublayer.deserialize(d['input_layer'])
		new_info.output_layer = LayerSublayer.deserialize(d['output_layer'])
		new_info.feature_type = FeatureType.deserialize(d['feature_type'])
		new_info.use_relu = d['use_relu']
		new_info.attn_head = d['attn_head']
		new_info.observable_tokens = d['observable_tokens']
		new_info.observable_weights = d['observable_weights']
		new_info.sae_info = SAEInfo.deserialize(d['sae_info']) if 'sae_info' in d and d['sae_info'] is not None else None
		new_info.feature_idx = d['feature_idx']

		return new_info
	
	def save(self, dirpath, json_filename="feature.json", tensors_filename="feature_tensors.safetensors", save_tensors=True, out_zipfile=None):
		# note: tensors_filename is relative to the same directory in which the json will be saved

		if out_zipfile is None:
			if not os.path.exists(dirpath):
				os.mkdir(dirpath)
		"""else:
			if not zipfile.Path(out_zipfile, dirpath).exists():
				out_zipfile.mkdir(dirpath)"""

		retdict = self.serialize()

		if save_tensors:
			tensors_dict = { 'encoder_vector': self.encoder_vector, 'decoder_vector': self.decoder_vector }
			if out_zipfile is None:
				safetensors.torch.save_file(tensors_dict, os.path.join(dirpath, tensors_filename))
			else:
				tensor_bytes = safetensors.torch.save(tensors_dict)
				with out_zipfile.open(os.path.join(dirpath, tensors_filename), "w") as ofp:
					ofp.write(tensor_bytes)
			retdict['tensors_filename'] = tensors_filename
		else:
			retdict['tensors_filename'] = None
	
		if out_zipfile is None:
			with open(os.path.join(dirpath, json_filename), "w") as ofp:
				json.dump(retdict, ofp)
		else:
			with out_zipfile.open(os.path.join(dirpath, json_filename), "w") as ofp:
				s = json.dumps(retdict)
				ofp.write(bytes(s, 'ascii'))

		return json_filename
	
	@classmethod
	def load(cls, json_path, load_tensors=True, in_zipfile=None):
		if in_zipfile is None:
			with open(json_path, "r") as fp:
				d = json.load(fp)
		else:
			with in_zipfile.open(json_path, "r") as fp:
				d = json.load(fp)

		new_info = cls.deserialize(d)

		tensors_filename = d['tensors_filename']
		if load_tensors and tensors_filename is not None:
			dirname = os.path.dirname(json_path)

			if in_zipfile is None:
				with safetensors.safe_open(os.path.join(dirname, tensors_filename), framework="pt", device=device) as tensors:
					new_info.encoder_vector = tensors.get_tensor('encoder_vector')
					new_info.decoder_vector = tensors.get_tensor('decoder_vector')
			else:
				with in_zipfile.open(os.path.join(dirname, tensors_filename), "r") as fp:
					tensor_bytes = fp.read()
				tensors = safetensors.torch.load(tensor_bytes)
				new_info.encoder_vector = tensors['encoder_vector'].to(device=device, dtype=dtype)
				new_info.decoder_vector = tensors['decoder_vector'].to(device=device, dtype=dtype)
					
		return new_info

	@classmethod
	@no_grad()
	def init_from_sae_feature(cls, sae_info : 'SAEInfo', feature_idx : int, name=None, description=None):
		feature = cls() 
		sae = sae_info.sae

		if name is None or name == '':
			feature.name = f'{sae_info.short_name}[{feature_idx}]'
		else: feature.name = name
		if description is not None: feature.description = description

		feature.encoder_vector = sae.W_enc[:, feature_idx].clone().detach().to(device=device, dtype=dtype)
		#feature.encoder_bias = (sae.b_enc[feature_idx] - (sae.b_dec @ sae.W_enc)[feature_idx]).item()
		feature.encoder_bias = sae.b_enc[feature_idx].item()
		feature.decoder_vector = sae.W_dec[feature_idx].clone().detach().to(device=device, dtype=dtype)

		feature.input_layer = sae_info.input_layer
		feature.output_layer = sae_info.output_layer

		feature.feature_type = FeatureType.SAE
		feature.use_relu = True

		feature.sae_info = sae_info
		feature.feature_idx = feature_idx

		return feature

	@classmethod
	@no_grad()
	def init_from_observable(cls, model : HookedTransformer, token_strs : List[str], weights : List[float], do_unembed_pullback : bool=False, name=None, description=None):
		feature = cls()
		if name is None or name == '':
			feature.name = 'observable'
		else: feature.name = name
		if description is not None: feature.description = description

		feature.observable_tokens = token_strs
		feature.observable_weights = weights

		observable = torch.zeros(model.W_U.shape[-1])
		for token_str, weight in zip(token_strs, weights):
			try:
				token_idx = model.to_single_token(token_str)
			except AssertionError:
				raise Exception(f"Invalid token: \"{token_str}\" is not a single token")
			observable[token_idx] = weight
		observable = observable.to(device=device, dtype=dtype)

		feature.encoder_bias = 0
		feature.output_layer = LayerSublayer(len(model.blocks), Sublayer.LOGITS)
		feature.decoder_vector = observable/(torch.linalg.norm(observable)**2)
		if not do_unembed_pullback:
			feature.encoder_vector = observable
			feature.input_layer = feature.output_layer
		else:
			feature.encoder_vector = model.W_U @ observable
			feature.input_layer = LayerSublayer(len(model.blocks), Sublayer.PRE_LOGITS)

		feature.feature_type = FeatureType.OBSERVABLE

		return feature

	@classmethod
	@no_grad()
	def init_from_attn_pullback(cls, model : HookedTransformer, base_feature : 'FeatureInfo', layer : int, head : int):
		feature = cls()

		feature.name = f'attn{layer}[{head}]'

		feature.encoder_bias = 0
		feature.input_layer = LayerSublayer(layer, Sublayer.ATTN_IN)
		feature.output_layer = LayerSublayer(layer, Sublayer.RESID_POST if model.cfg.parallel_attn_mlp else Sublayer.RESID_MID)
		feature.decoder_vector = base_feature.encoder_vector/(torch.linalg.norm(base_feature.encoder_vector)**2)
		feature.encoder_vector = model.OV[layer, head] @ base_feature.encoder_vector

		feature.attn_head = head

		feature.feature_type = FeatureType.PULLBACK

		return feature

	@classmethod
	@no_grad()
	def init_from_vector(cls, model : HookedTransformer, layer : LayerSublayer, vector, name=None, contravariant=True):
		feature = cls()

		if name is None:
			feature.name = f'{layer}_vector'
		else:
			feature.name = name

		feature.encoder_bias = 0
		feature.input_layer = layer
		feature.output_layer = layer

		if contravariant:
			feature.encoder_vector = vector/(torch.linalg.norm(vector)**2)
			feature.decoder_vector = vector
		else:
			feature.encoder_vector = vector
			feature.decoder_vector = vector/(torch.linalg.norm(vector)**2)

		feature.feature_type = FeatureType.OTHER

		return feature

	@no_grad()
	def get_activs(self, tensor, use_relu=None):
		# tensor: [batch, tokens, dim]
		if tensor.dtype != self.encoder_vector.dtype:
			# hack to take care of transformerlens forcibly storing layernorm activations as float32
			tensor = tensor.to(dtype=self.encoder_vector.dtype)
		pre_acts = torch.einsum('d, ...d -> ...', self.encoder_vector, tensor)
		acts = pre_acts + self.encoder_bias
		if use_relu is None: use_relu = self.use_relu
		if use_relu: acts = torch.nn.functional.relu(acts)
		
		return _to_numpy(acts)

	@no_grad()
	def get_deembeddings(self, model : HookedTransformer):
		return _to_numpy(model.W_E @ self.encoder_vector)

### Computational=path-related classes/methods ###

@dataclass
class AttribInfo:
	feature_info : Optional[FeatureInfo] = None
	token_pos : Optional[int] = None

	invar_factor : float = 1.0
	ln_constant : float = 1.0
	attn_factor : float = 1.0
	feature_activ : float = 1.0

	total_invar_factor : Optional[float] = None
	total_ln_constant : Optional[float] = None
	total_attn_factor : Optional[float] = None
	total_attrib : Optional[float] = None

	parent_ln_constant : Optional[float] = None
	total_parent_ln_constant : Optional[float] = None

	top_child_components : Optional[List['ComponentInfo']] = None
	top_child_contribs : Optional[List[float]] = None

	name : Optional[str] = None
	description : Optional[str] = None

	# for use with steering vectors
	# is_unsteered_attrib : bool = False
	unsteered_attrib : Optional['AttribInfo'] = None

	def serialize_base(self):
		return	{
			'token_pos': self.token_pos,

			'invar_factor': self.invar_factor,
			'ln_constant': self.ln_constant,
			'attn_factor': self.attn_factor,
			'feature_activ': self.feature_activ,

			'parent_ln_constant': self.parent_ln_constant,
			'total_parent_ln_constant': self.total_parent_ln_constant,

			'total_invar_factor': self.total_invar_factor,
			'total_ln_constant': self.total_ln_constant,
			'total_attn_factor': self.total_attn_factor,
			'total_attrib': self.total_attrib,

			'unsteered_attrib': None if self.unsteered_attrib is None else self.unsteered_attrib.serialize()

		}

	def serialize(self):
		retdict = self.serialize_base()
		retdict = {**retdict, 
			'top_child_components': [x.serialize() for x in self.top_child_components] if self.top_child_components is not None else None,
			'top_child_contribs': self.top_child_contribs,

			'name': self.name,
			'description': self.description
		}
		retdict['feature_info'] = self.feature_info.serialize()
		return retdict
	
	def save(self, dirpath, json_filename="attrib.json", save_tensors=True, out_zipfile=None):
		if out_zipfile is None:
			if not os.path.exists(dirpath):
				os.mkdir(dirpath)
		"""else:
			if not zipfile.Path(out_zipfile, dirpath).exists():
				out_zipfile.mkdir(dirpath)"""

		retdict = self.serialize()

		# save feature info
		retdict['feature_json_filename'] = self.feature_info.save(dirpath, save_tensors=save_tensors, out_zipfile=out_zipfile)

		if out_zipfile is None:
			with open(os.path.join(dirpath, json_filename), "w") as ofp:
				json.dump(retdict, ofp)
		else:
			with out_zipfile.open(os.path.join(dirpath, json_filename), "w") as ofp:
				s = json.dumps(retdict)
				ofp.write(bytes(s, 'ascii'))

		return json_filename 

	@classmethod
	def deserialize(cls, d):
		new_info = cls(
			feature_info=FeatureInfo.deserialize(d['feature_info']),
			token_pos=d['token_pos'],

			invar_factor=d['invar_factor'],
			ln_constant=d['ln_constant'],
			attn_factor=d['attn_factor'],
			feature_activ=d['feature_activ'],

			total_invar_factor=d['total_invar_factor'],
			total_ln_constant=d['total_ln_constant'],
			total_attn_factor=d['total_attn_factor'],
			total_attrib=d['total_attrib'],

			total_parent_ln_constant=d['total_parent_ln_constant'] if 'total_parent_ln_constant' in d else None,
			parent_ln_constant=d['parent_ln_constant'] if 'parent_ln_constant' in d else None,

			top_child_components=[ComponentInfo.deserialize(x) for x in d['top_child_components']] if 'top_child_components' in d and d['top_child_components'] is not None else None,
			top_child_contribs=d['top_child_contribs'] if 'top_child_contribs' in d else None,

			name=d['name'] if 'name' in d else None,
			description=d['description'] if 'description' in d else None,

			unsteered_attrib=cls.deserialize(d['unsteered_attrib']) if 'unsteered_attrib' in d and d['unsteered_attrib'] is not None else None,
		)
		return new_info
	
	@classmethod
	def load(cls, json_path, load_tensors=True, in_zipfile=None):
		if in_zipfile is None:
			with open(json_path, "r") as fp:
				d = json.load(fp)
		else:
			with in_zipfile.open(json_path, "r") as fp:
				d = json.load(fp)
		new_info = cls.deserialize(d)

		feature_info_filename = d['feature_json_filename']
		dirname = os.path.dirname(json_path)
		new_info.feature_info = FeatureInfo.load(os.path.join(dirname, feature_info_filename), load_tensors=load_tensors, in_zipfile=in_zipfile)
		return new_info

class ComponentType(enum.Enum):
	SAE_FEATURE = 0
	ATTN_HEAD = 1
	EMBED = 2

	def serialize(self):
		return self.name.lower()

	@classmethod
	def deserialize(cls, s):
		return {x.name.lower(): x for x in cls}[s]

@dataclass
class ComponentInfo:
	component_type : ComponentType
	token_pos : int

	attn_head : Optional[int] = None
	attn_layer : Optional[int] = None
	sae_idx : Optional[int] = None
	feature_idx : Optional[int] = None
	embed_vocab_idx : Optional[int] = None

	def serialize(self):
		retdict = {
			"component_type": self.component_type.serialize(),
			"token_pos": self.token_pos,

			"attn_head": self.attn_head,
			"attn_layer": self.attn_layer,
			"sae_idx": self.sae_idx,
			"feature_idx": self.feature_idx,
			"embed_vocab_idx": self.embed_vocab_idx
		}
		return retdict
	
	@classmethod
	def deserialize(cls, d):
		new_info = cls(
			component_type=ComponentType.deserialize(d['component_type']),
			token_pos=d['token_pos'],

			attn_head=d['attn_head'],
			attn_layer=d['attn_layer'],
			sae_idx=d['sae_idx'],
			feature_idx=d['feature_idx'],
			embed_vocab_idx=d['embed_vocab_idx']
		)
		return new_info

class ComputationalPath:
	def __init__(self, name : str, description : str = ""):
		self.name = name
		self.description = description
		self.nodes : List[AttribInfo] = []
		self.is_outdated : bool = False
		self.outdated_token_strs : Optional[List[str]] = None

	def save(self, dirpath, json_filename="path.json", save_tensors=True, out_zipfile=None):
		if out_zipfile is None:
			if not os.path.exists(dirpath):
				os.mkdir(dirpath)

		retdict = {
			'name': self.name,
			'description': self.description,
			'node_json_filenames': [],
			'is_outdated': self.is_outdated,
			'outdated_token_strs': self.outdated_token_strs
		}

		for i, node in enumerate(self.get_total_attribs()):
			node_dir_rel_path = f'node_{i}'
			attrib_json_filename = node.save(os.path.join(dirpath, node_dir_rel_path), save_tensors=save_tensors, out_zipfile=out_zipfile)
			retdict['node_json_filenames'].append(
				os.path.join(node_dir_rel_path, attrib_json_filename)
			)

		if out_zipfile is None:
			with open(os.path.join(dirpath, json_filename), "w") as ofp:
				json.dump(retdict, ofp)
		else:
			with out_zipfile.open(os.path.join(dirpath, json_filename), "w") as ofp:
				s = json.dumps(retdict)
				ofp.write(bytes(s, 'ascii'))

		return json_filename

	@classmethod
	def load(cls, json_path, load_tensors=True, in_zipfile=None):
		if in_zipfile is None:
			with open(json_path, "r") as fp:
				d = json.load(fp)
		else:
			with in_zipfile.open(json_path, "r") as fp:
				d = json.load(fp)

		new_info = cls(name=d['name'], description=d['description'])
		new_info.is_outdated = d['is_outdated']
		new_info.outdated_token_strs = d['outdated_token_strs']

		dirname = os.path.dirname(json_path)
		for node_json_filename in d['node_json_filenames']:
			new_node_info = AttribInfo.load(os.path.join(dirname, node_json_filename), load_tensors=load_tensors, in_zipfile=in_zipfile)
			new_info.nodes.append(new_node_info)
		return new_info
	
	def _get_total_attribs_for_node_list(self, node_list):
		root = node_list[0]
		if root.total_attrib is None:
			retlist = [AttribInfo(
				feature_info = root.feature_info,
				token_pos = root.token_pos,

				invar_factor = root.invar_factor,
				ln_constant = root.ln_constant,
				attn_factor = root.attn_factor,
				feature_activ = root.feature_activ,

				total_invar_factor = root.invar_factor,
				total_ln_constant = root.ln_constant,
				total_attn_factor = root.attn_factor,

				top_child_components = root.top_child_components,
				top_child_contribs = root.top_child_contribs,

				name = root.name if root.name is not None else root.feature_info.name
			)]
			retlist[0].total_attrib = retlist[0].invar_factor * retlist[0].attn_factor * retlist[0].feature_activ
		else:
			retlist = [root]

		retlist[0].parent_ln_constant = 1.0
		retlist[0].total_parent_ln_constant = 1.0

		for i in range(1, len(node_list)):
			cur_node = node_list[i]
			if cur_node.total_attrib is not None:
				retlist.append(cur_node)
				continue

			prev_node = retlist[i-1]

			cur_node.total_invar_factor = cur_node.invar_factor * prev_node.total_invar_factor
			cur_node.total_ln_constant = cur_node.ln_constant * prev_node.total_ln_constant
			cur_node.total_attn_factor = cur_node.attn_factor * prev_node.total_attn_factor

			cur_node.parent_ln_constant = prev_node.ln_constant
			cur_node.total_parent_ln_constant = prev_node.total_ln_constant

			# IMPORTANT: note that we use the PREVIOUS NODE's layernorm constant to compute the current node's attrib
			cur_node.total_attrib = cur_node.total_invar_factor * prev_node.total_ln_constant * cur_node.total_attn_factor * cur_node.feature_activ

			if cur_node.name is None: cur_node.name = cur_node.feature_info.name

			retlist.append(cur_node)
		return retlist

	def get_total_attribs(self):
		if len(self.nodes) == 0: return []
		retlist = self._get_total_attribs_for_node_list(self.nodes)
		for original_node, new_node in zip(self.nodes, retlist):
			new_node.unsteered_attrib = original_node.unsteered_attrib
		self.nodes = retlist

		# now, do the same for each node's unsteered_attrib
		# ASSUMPTION: either all unsteered_attribs are None or no unsteered_attribs are None
		unsteered_attribs = []
		unsteered_attrib_none = False
		for node in self.nodes:
			if node.unsteered_attrib is None:
				unsteered_attrib_none = True
				break
			unsteered_attribs.append(node.unsteered_attrib)
		
		if not unsteered_attrib_none:
			unsteered_attribs = self._get_total_attribs_for_node_list(unsteered_attribs)

			for node, unsteered_attrib in zip(self.nodes, unsteered_attribs):
				node.unsteered_attrib = unsteered_attrib

		return retlist

### SAE-related classes/methods ###

# the SAEConfig and SAE classes are inspired by the SAELens classes
# but these classes are trimmed down, since we don't focus on training SAEs

# the SAEConfig class only contains info essential to performing inference
#  with a given SAE
# (in contrast, the SAEInfo class contains auxiliary metadata related to
#  whence one procures an SAE)
@dataclass
class SAEConfig:
	d_in : int
	num_features : int

	d_out : Optional[int] = None
	dtype : torch.dtype = torch.float32
	act_fn : str = "relu"

	top_k : Optional[int] = None

	def __post_init__(self):
		if self.d_out is None: self.d_out = self.d_in

	def serialize(self):
		d = dataclasses.asdict(self)
		d['dtype'] = str(d['dtype'])
		return d
	
	@classmethod
	def deserialize(cls, d):
		return cls(
			d_in=d['d_in'],
			num_features=d['num_features'],
			d_out=d.get('d_out', None),
			dtype=DTYPE_DICT[d.get('dtype', None)],
			act_fn=d.get('act_fn', 'relu'),

			top_k=d.get('top_k', None)
		)
	
if backend_import_cfg.import_expensive_modules:
	sae_superclass = torch.nn.Module
else:
	sae_superclass = object

class SAE(sae_superclass):
	def __init__(self, cfg : SAEConfig, load_tensors=True):	
		super().__init__()
		self.cfg = cfg

		if load_tensors and backend_import_cfg.import_expensive_modules:
			self.W_enc = torch.nn.Parameter(
				torch.empty(
					self.cfg.d_in, self.cfg.num_features, dtype=self.cfg.dtype
				)
			)

			self.b_enc = torch.nn.Parameter(
				torch.empty(
					self.cfg.num_features, dtype=self.cfg.dtype
				)
			)

			self.W_dec = torch.nn.Parameter(
				torch.empty(
					self.cfg.num_features, self.cfg.d_out, dtype=self.cfg.dtype
				)
			)

			self.b_dec = torch.nn.Parameter(
				torch.empty(
					self.cfg.d_out, dtype=self.cfg.dtype
				)
			)

		act_fns = {
			'relu': torch.nn.functional.relu,
			'id': lambda x: x,
			'top_k': lambda x: self._top_k(x)
		}
		self.act_fn = act_fns[self.cfg.act_fn]
	
	def _top_k(self, x):
		acts = torch.zeros(x.shape, dtype=x.dtype).to(device=x.device)
		vals, idxs = torch.topk(x, k=self.cfg.top_k)
		return torch.scatter(acts, -1, idxs, vals)
	
	def get_activs(self, x):
		if x.dtype != self.W_enc.dtype:
			# hack to take care of transformerlens forcibly storing layernorm activations as float32
			x = x.to(dtype=self.W_enc.dtype)
		pre_acts = torch.einsum('df, ...d -> ...f', self.W_enc, x)
		acts = self.act_fn(pre_acts + self.b_enc)
		return acts
	
	def forward(self, x):
		acts = self.get_activs(x)
		post_acts = torch.einsum('fd, ...f -> ...d', self.W_dec, acts)
		outs = post_acts + self.b_dec
		return outs
	
	def save(self, dirpath, json_filename="sae.json", save_tensors=False, tensors_filename=None, out_zipfile=None):
		if tensors_filename is None:
			tensors_filename = f"{os.path.splitext(json_filename)[0]}.safetensors"
		
		cfg_dict = self.cfg.serialize()
		
		if out_zipfile is None:
			if not os.path.exists(dirpath):
				os.mkdir(dirpath)
		"""else:
			if not zipfile.Path(out_zipfile, dirpath).exists():
				out_zipfile.mkdir(dirpath)"""
		if save_tensors:
			tensors_dict = { 'W_enc': self.W_enc, 'W_dec': self.W_dec, 'b_enc': self.b_enc, 'b_dec': self.b_dec }
			if out_zipfile is None:
				safetensors.torch.save_file(tensors_dict, os.path.join(dirpath, tensors_filename))
			else:
				tensor_bytes = safetensors.torch.save(tensors_dict)
				with out_zipfile.open(os.path.join(dirpath, tensors_filename), "w") as ofp:
					ofp.write(tensor_bytes)
			cfg_dict['tensors_filename'] = tensors_filename
		else:
			cfg_dict['tensors_filename'] = None

		if out_zipfile is None:
			with open(os.path.join(dirpath, json_filename), "w") as ofp:
				json.dump(cfg_dict, ofp)
		else:
			with out_zipfile.open(os.path.join(dirpath, json_filename), "w") as ofp:
				s = json.dumps(cfg_dict)
				ofp.write(bytes(s, 'ascii'))
		return json_filename

	def load_tensors(self, tensors_path, in_zipfile=None):
		if in_zipfile is None:
			with safetensors.safe_open(tensors_path, framework="pt", device="cpu") as tensors:
				self.W_enc.data = tensors.get_tensor('W_enc')
				self.W_dec.data = tensors.get_tensor('W_dec')
				self.b_enc.data = tensors.get_tensor('b_enc')
				self.b_dec.data = tensors.get_tensor('b_dec')
		else:
			with in_zipfile.open(tensors_path, "r") as fp:
				tensor_bytes = fp.read()
			tensors = safetensors.torch.load(tensor_bytes)
			self.W_enc.data = tensors['W_enc']
			self.W_dec.data = tensors['W_dec']
			self.b_enc.data = tensors['b_enc']
			self.b_dec.data = tensors['b_dec']
	
	@classmethod
	@no_grad()
	def load(cls, json_path, load_tensors=True, in_zipfile=None):
		if in_zipfile is None:
			with open(json_path, "r") as fp:
				d = json.load(fp)
		else:
			with in_zipfile.open(json_path, "r") as fp:
				d = json.load(fp)

		cfg = SAEConfig.deserialize(d)
		new_sae = cls(cfg)

		tensors_filename = d['tensors_filename']
		if load_tensors and tensors_filename is not None:
			dirname = os.path.dirname(json_path)
			tensors_path = os.path.join(dirname, tensors_filename)
			new_sae.load_tensors(tensors_path, in_zipfile)

		return new_sae

class SAEInfo:
	def serialize(self):
		return {
			'name': self.name,
			'input_layer': self.input_layer.serialize(),
			'output_layer': self.output_layer.serialize(),
			'short_name': self.short_name,
			'model_path': self.model_path,

			'upstreams': [upstream.serialize() for upstream in self.upstreams]
		}
	def save(self, dirpath, json_filename="sae_info.json", sae_filename='sae.json', save_tensors=False, tensors_filename=None, out_zipfile=None):
		retdict = self.serialize()
		retdict['sae_filename'] = sae_filename

		if out_zipfile is None:
			if not os.path.exists(dirpath):
				os.mkdir(dirpath)
		"""else:
			if not zipfile.Path(out_zipfile, dirpath).exists():
				out_zipfile.mkdir(dirpath)"""

		self.sae.save(dirpath, json_filename=sae_filename, save_tensors=save_tensors, tensors_filename=tensors_filename, out_zipfile=out_zipfile)

		if out_zipfile is None:
			with open(os.path.join(dirpath, json_filename), "w") as ofp:
				json.dump(retdict, ofp)
		else:
			with out_zipfile.open(os.path.join(dirpath, json_filename), "w") as ofp:
				s = json.dumps(retdict)
				ofp.write(bytes(s, 'ascii'))
		return json_filename

	@classmethod
	def _try_load_from_upstream(cls, upstream, load_tensors=True):
		# get SAEInfo dict
		sae_info_json_path = upstream.get_path()
		if sae_info_json_path is None: return None
		with open(sae_info_json_path, "r") as fp:
			d = json.load(fp)
		
		# get SAE dict
		sae_filename = d.get("sae_filename", None)
		if sae_filename is None: return None
		sae_json_datapath = upstream.get_sibling_datapath(sae_filename)
		sae_json_path = sae_json_datapath.get_path()
		if sae_json_path is None: return None

		# get tensors
		if load_tensors:
			with open(sae_json_path, "r") as fp:
				sae_cfg_dict = json.load(fp)
			tensors_filename = sae_cfg_dict.get('tensors_filename', None)
			if tensors_filename is None: return None
			sae_tensors_datapath = sae_json_datapath.get_sibling_datapath(tensors_filename)
			sae_tensors_path = sae_tensors_datapath.get_path()
			if sae_tensors_path is None: return None

		# at this point, we've confirmed that every file that needs to 
		#  be loaded actually exists
		# and if our upstream is on HuggingFace, then we've also downloaded
		#  everything that needs to be downloaded
		# ergo, 'tis now time to load
		# (note: this works because hf cache preserves directory structure) 
		try:
			retval = cls.load(sae_info_json_path, load_tensors=load_tensors)
			return retval
		except:
			return None

	def load_from_upstreams(self, load_tensors=True):
		for upstream in self.upstreams:
			try:
				retval = type(self)._try_load_from_upstream(upstream, load_tensors=load_tensors)
			except Exception as e:
				raise e
				retval = None
			if retval is not None: return retval
		return None

	@classmethod
	def load_from_hf(cls, hf_repo_id, hf_json_path, load_tensors=True):
		return cls._try_load_from_upstream(DataPath(path_type=DataPathType.HUGGINGFACE, hf_repo_id=hf_repo_id, hf_path=hf_json_path), load_tensors=load_tensors)
	
	@classmethod
	def deserialize(cls, d):
		sae_info = cls()
		sae_info.name = d.get('name', '')
		sae_info.input_layer = LayerSublayer.deserialize(d['input_layer'])
		if (output_layer_val := d.get('output_layer', None)) is not None:
			sae_info.output_layer = LayerSublayer.deserialize(output_layer_val)
		else:
			sae_info.output_layer = sae_info.input_layer

		if (short_name := d.get('short_name', None)) is not None and short_name != "":
			sae_info.short_name = short_name
		else:
			if sae_info.input_layer == sae_info.output_layer:
				sae_info.short_name = f'sae{sae_info.input_layer}'
			else:
				if sae_info.input_layer.layer == sae_info.output_layer.layer:
					sae_info.short_name = f'tc{sae_info.input_layer}'
				else:
					sae_info.short_name = f'tc{sae_info.input_layer}_{sae_info.output_layer}'

		sae_info.model_path = d.get('model_path', None)
		sae_info.upstreams = [DataPath.deserialize(upstream) for upstream in d.get('upstreams', None)]
		return sae_info

	@classmethod
	def load(cls, json_path, load_tensors=True, in_zipfile=None):
		if in_zipfile is None:
			with open(json_path, "r") as fp:
				d = json.load(fp)
		else:
			with in_zipfile.open(json_path, "r") as fp:
				d = json.load(fp)

		sae_info = cls.deserialize(d)

		sae_info.upstreams = [DataPath.deserialize(upstream) for upstream in d.get('upstreams', None)]
		
		dirname = os.path.dirname(json_path)
		sae_json_path = os.path.join(dirname, d['sae_filename']) 
		sae_info.sae = SAE.load(sae_json_path, load_tensors=load_tensors, in_zipfile=in_zipfile)
		sae_info.sae.to(dtype=dtype)

		return sae_info

	# now for some actual circuit analysis methods
	@no_grad()
	def get_feature_input_invariant_scores(self, feature : FeatureInfo):
		return _to_numpy(self.sae.W_dec.to(device=device, dtype=dtype) @ feature.encoder_vector)

	@no_grad()
	def get_feature_cossims(self, feature : FeatureInfo):
		normed_W_dec = torch.einsum('ij,i->ij', self.sae.W_dec, torch.linalg.norm(self.sae.W_dec, dim=1))
		normed_vector = feature.encoder_vector / torch.linalg.norm(feature.encoder_vector)
		return _to_numpy(normed_W_dec @ normed_vector)

	def __str__(self):
		return self.short_name
	
class DataPathType(enum.Enum):
	HUGGINGFACE = 'huggingface'
	RELATIVE = 'relative_path'
	ABSOLUTE = 'absolute_path'

	def serialize(self):
		return self.value
	@classmethod
	def deserialize(cls, s):
		return {x.value: x for x in cls}[s]

@dataclass
class DataPath:
	path_type : DataPathType
	hf_repo_id : Optional[str] = None
	hf_path : Optional[str] = None

	absolute_path : Optional[str] = None

	relative_path : Optional[str] = None

	def serialize(self):
		d = dataclasses.asdict(self)
		d['path_type'] = d['path_type'].serialize()
		return d
	
	@classmethod
	def deserialize(cls, d):
		return cls(
			path_type = DataPathType.deserialize(d['path_type']),
			hf_repo_id = d.get('hf_repo_id', None),
			hf_path = d.get('hf_path', None),

			absolute_path = d.get('absolute_path', None),

			relative_path = d.get('relative_path', None)
		)

	def get_path(self):
		if self.path_type == DataPathType.RELATIVE:
			if os.path.exists(self.relative_path): return self.relative_path
			else: return None

		if self.path_type == DataPathType.ABSOLUTE:
			if os.path.exists(self.absolute_path): return self.absolute_path
			else: return None

		if self.path_type == DataPathType.HUGGINGFACE:
			path = hf_hub.try_to_load_from_cache(self.hf_repo_id, self.hf_path)
			if path is None and path != hf_hub._CACHED_NO_EXIST:
				hf_hub.hf_hub_download(self.hf_repo_id, self.hf_path)
				path = hf_hub.try_to_load_from_cache(self.hf_repo_id, self.hf_path)
			return path if path != hf_hub._CACHED_NO_EXIST else None
	
	def get_sibling_datapath(self, filename):
		if self.path_type == DataPathType.RELATIVE:
			new_path = os.path.join(os.path.dirname(self.relative_path), filename)
			return DataPath(path_type=DataPathType.RELATIVE, relative_path=new_path)

		if self.path_type == DataPathType.ABSOLUTE:
			new_path = os.path.join(os.path.dirname(self.absolute_path), filename)
			return DataPath(path_type=DataPathType.ABSOLUTE, absolute_path=new_path)

		if self.path_type == DataPathType.HUGGINGFACE:
			new_hf_path = os.path.join(os.path.dirname(self.hf_path), filename)
			return DataPath(path_type=DataPathType.HUGGINGFACE, hf_repo_id=self.hf_repo_id, hf_path=new_hf_path)

### SteeringVector class ###
# Used for steering prompts

@dataclass
class SteeringVector:
	feature_info : FeatureInfo
	token_pos : Optional[int]

	coefficient : float = 1.0
	do_clamp : bool = True
	use_encoder : bool = False

	name : Optional[str] = None
	description : Optional[str] = None

	def serialize(self):
		return {
			'name': self.name,
			'description': self.description,

			'token_pos': self.token_pos,
			'coefficient': self.coefficient,
			'do_clamp': self.do_clamp,

			'feature_info': self.feature_info.serialize()
		}
	
	def save(self, dirpath, json_filename="steering_vector.json", save_tensors=True, out_zipfile=None):
		if out_zipfile is None:
			if not os.path.exists(dirpath):
				os.mkdir(dirpath)

		retdict = self.serialize()

		# save feature info
		retdict['feature_json_filename'] = self.feature_info.save(dirpath, save_tensors=save_tensors, out_zipfile=out_zipfile)

		if out_zipfile is None:
			with open(os.path.join(dirpath, json_filename), "w") as ofp:
				json.dump(retdict, ofp)
		else:
			with out_zipfile.open(os.path.join(dirpath, json_filename), "w") as ofp:
				s = json.dumps(retdict)
				ofp.write(bytes(s, 'ascii'))

		return json_filename
	
	@classmethod
	def deserialize(cls, d):
		return cls(
			feature_info=FeatureInfo.deserialize(d['feature_info']),
			token_pos=d['token_pos'],

			coefficient=d['coefficient'],
			do_clamp=d['do_clamp'],

			name=d['name'] if 'name' in d else None,
			description=d['description'] if 'description' in d else None
		)
	
	@classmethod
	def load(cls, json_path, load_tensors=True, in_zipfile=None):
		if in_zipfile is None:
			with open(json_path, "r") as fp:
				d = json.load(fp)
		else:
			with in_zipfile.open(json_path, "r") as fp:
				d = json.load(fp)
		new_info = cls.deserialize(d)

		feature_info_filename = d['feature_json_filename']
		dirname = os.path.dirname(json_path)
		new_info.feature_info = FeatureInfo.load(os.path.join(dirname, feature_info_filename), load_tensors=load_tensors, in_zipfile=in_zipfile)
		return new_info

	def make_hooks(self):
		hooks = []
		feature_activ = 0.0
		def in_hook(hidden_state, hook):
			if self.do_clamp:
				nonlocal feature_activ
				if self.token_pos is not None:
					feature_activ = self.feature_info.get_activs(
						hidden_state[0, self.token_pos] #TODO: allow for batching?
					)
				else:
					feature_activ = self.feature_info.get_activs(
						hidden_state[0] #TODO: allow for batching?
					)
		def out_hook(hidden_state, hook):
			vector = self.feature_info.decoder_vector if not self.use_encoder else self.feature_info.encoder_vector
			if self.token_pos is not None:
				hidden_state[:, self.token_pos] += vector * (self.coefficient - feature_activ)
			else:
				if self.do_clamp:
					hidden_state[0] += torch.einsum('d, t -> td', vector, self.coefficient - torch.from_numpy(feature_activ).to(device=device, dtype=dtype))
				else:
					hidden_state[0] += vector * self.coefficient
			return hidden_state

		return [
			(self.feature_info.input_layer.to_hookpoint_str(), in_hook),
			(self.feature_info.output_layer.to_hookpoint_str(), out_hook)
		]

###

# below, a useful class for "lists" (really dicts) that automatically assign ids when adding members
# this will be used for lists of feature vectors, lists of SAEs, and lists of prompts

class IdDict:
	def __init__(self, name="", description=""):
		self.lock = RLock() # used to atomically increment current id
		self.cur_id = 0
		self.dict = {}

		self.name = name
		self.description = description

	def add(self, item):
		with self.lock:
			new_id = self.cur_id
			self.dict[self.cur_id] = item
			self.cur_id += 1
		return new_id

	def remove(self, idx):
		with self.lock:
			del self.dict[idx]


### Prompt class ###
# This is where the real magic happens; most circuit analysis methods are implemented here
class Prompt:
	def __init__(self, name="", description=""):
		self.name = name
		self.description = description

		self.lock = RLock()

		self.tokens : List[str] = [] # list of tokens for prompt
		self.cache = None # cached model activations
		self.logits = None # cached logits

		self.comp_paths = IdDict() # list of computational paths 
		self.cur_comp_path = ComputationalPath(name="Current path") # the current working/tmp computational path

		self.cur_feature_list_idx = None # the current working feature list

		self.steering_vectors = IdDict() # list of active steering vectors
		# if steering was applied, then this dict contains the original model activations before steering
		self.unsteered_cache = None
		# same with logits
		self.unsteered_logits = None


	def save(self, dirpath, json_filename="prompt.json", save_tensors=True, out_zipfile=None): 
		if out_zipfile is None:
			if not os.path.exists(dirpath):
				os.mkdir(dirpath)

		retdict = {
			'name': self.name,
			'description': self.description,
			'tokens': self.tokens,
			'comp_path_json_filenames': {},
			'steering_vector_json_filenames': {},
			'cur_feature_list_idx': self.cur_feature_list_idx,
		}

		cur_comp_path_dirname = 'cur_comp_path'
		cur_comp_path_json_filename = self.cur_comp_path.save(os.path.join(dirpath, cur_comp_path_dirname), save_tensors=save_tensors, out_zipfile=out_zipfile)
		retdict['cur_comp_path_json_filename'] = os.path.join(cur_comp_path_dirname, cur_comp_path_json_filename)

		with self.comp_paths.lock:
			for idx, comp_path in self.comp_paths.dict.items():
				comp_path_dirname = f'comp_path_{idx}'
				comp_path_json_filename = comp_path.save(os.path.join(dirpath, comp_path_dirname), save_tensors=save_tensors, out_zipfile=out_zipfile)
				retdict['comp_path_json_filenames'][idx] = os.path.join(comp_path_dirname, comp_path_json_filename)

		with self.steering_vectors.lock:
			for idx, steering_vector in self.steering_vectors.dict.items():
				steering_vector_dirname = f'steering_vector_{idx}'
				steering_vector_json_filename = steering_vector.save(os.path.join(dirpath, steering_vector_dirname), save_tensors=save_tensors, out_zipfile=out_zipfile)
				retdict['steering_vector_json_filenames'][idx] = os.path.join(steering_vector_dirname, steering_vector_json_filename)

		if out_zipfile is None:
			with open(os.path.join(dirpath, json_filename), "w") as ofp:
				json.dump(retdict, ofp)
		else:
			with out_zipfile.open(os.path.join(dirpath, json_filename), "w") as ofp:
				s = json.dumps(retdict)
				ofp.write(bytes(s, 'ascii'))

		return json_filename

	@classmethod
	def load(cls, json_path, load_tensors=True, in_zipfile=None):
		if in_zipfile is None:
			with open(json_path, "r") as fp:
				d = json.load(fp)
		else:
			with in_zipfile.open(json_path, "r") as fp:
				d = json.load(fp) 

		new_info = cls(name=d['name'], description=d['description'])
		new_info.tokens = d['tokens']
		new_info.cur_feature_list_idx = d['cur_feature_list_idx']

		dirname = os.path.dirname(json_path)
		for idx, comp_path_json_filename in d['comp_path_json_filenames'].items():
			comp_path = ComputationalPath.load(os.path.join(dirname, comp_path_json_filename), load_tensors=load_tensors, in_zipfile=in_zipfile)
			new_info.comp_paths.add(comp_path)
		new_info.cur_comp_path = ComputationalPath.load(os.path.join(dirname, d['cur_comp_path_json_filename']), load_tensors=load_tensors, in_zipfile=in_zipfile)

		if 'steering_vector_json_filenames' in d:
			for idx, steering_vector_json_filename in d['steering_vector_json_filenames'].items():
				steering_vector = SteeringVector.load(os.path.join(dirname, steering_vector_json_filename), load_tensors=load_tensors, in_zipfile=in_zipfile)
				new_info.steering_vectors.add(steering_vector)

		return new_info

	@no_grad()
	def run_model_on_text(self, model : HookedTransformer, text : str):
		# here's how invalidation works:
		# 1. See if any steering vectors are invalidated. A steering vector is invalidated if it applies to a token that will be different in the new prompt (or comes after a token that will be different in the new prompt).
		# 2. If any steering vectors are invalidated, then remove the invalidated ones and rerun the whole prompt with the remaining steering vectors.
		# 3. Then, see if any computational paths are invalidated. A computational path is invalidated when it involves a token that is different in the new prompt (or comes after a token that will be different in the new prompt.)

		new_tokens = model.to_str_tokens(text)
		# idempotency
		if self.tokens == new_tokens and self.cache is not None: return self.tokens

		with self.lock:
			# get index of first different token
			minlen = min(len(self.tokens), len(new_tokens))
			diff_idx = None
			for i in range(minlen):
				if self.tokens[i] != new_tokens[i]:
					diff_idx = i
					break
			if diff_idx is None: diff_idx = minlen

			# invalidate steering vectors
			new_steering_vectors_dict = {idx: vector for idx, vector in self.steering_vectors.dict.items() if vector.token_pos is None or vector.token_pos < diff_idx}
			are_steering_vectors_invalidated = len(new_steering_vectors_dict) != len(self.steering_vectors.dict)
			self.steering_vectors.dict = new_steering_vectors_dict	

			# invalidate computational paths
			for path in self.comp_paths.dict.values():
				if not path.is_outdated and path.nodes[0].token_pos >= diff_idx:
					path.is_outdated = True
					path.outdated_token_strs = self.tokens
			if not self.cur_comp_path.is_outdated and len(self.cur_comp_path.nodes) > 0 and self.cur_comp_path.nodes[0].token_pos >= diff_idx:
				self.cur_comp_path.is_outdated = True
				self.cur_comp_path.outdated_token_strs = self.tokens

			# run the model
			logits, cache = model.run_with_cache(text, return_type='logits')
			self.tokens = new_tokens

			# cache management for steering vectors
			self.logits = logits
			self.cache = cache
			if len(self.steering_vectors.dict) == 0:
				self.unsteered_logits = None
				self.unsteered_cache = None
			else:
				self.unsteered_logits = logits
				self.unsteered_cache = cache

				self.run_with_steering_vectors(model)
	
		return new_tokens
	
	@no_grad()
	def run_with_steering_vectors(self, model : HookedTransformer):
		with self.lock:
			if len(self.steering_vectors.dict) == 0:
				self.cache = self.unsteered_cache
				self.logits = self.unsteered_logits
				self.unsteered_logits = None
				self.unsteered_cache = None
			else:
				hooks = []
				for steering_vector in self.steering_vectors.dict.values():
					hooks.extend(steering_vector.make_hooks())

				with model.hooks(fwd_hooks=hooks):
					logits, cache = model.run_with_cache("".join(self.tokens), return_type='logits', prepend_bos=False)

				if self.unsteered_logits is None:
					self.unsteered_logits = self.logits
				self.logits = logits

				if self.unsteered_cache is None:
					self.unsteered_cache = self.cache
				self.cache = cache

			# update computational paths
			for idx, comp_path in self.comp_paths.dict.items():
				for node_idx, node in enumerate(comp_path.nodes):
					prev_node = comp_path.nodes[node_idx-1] if node_idx > 0 else None
					comp_path.nodes[node_idx] = self.populate_attrib_info(node, prev_node)
			for node_idx, node in enumerate(self.cur_comp_path.nodes):
				prev_node = self.cur_comp_path.nodes[node_idx-1] if node_idx > 0 else None
				self.cur_comp_path.nodes[node_idx] = self.populate_attrib_info(node, prev_node)

	@no_grad()
	def get_feature_activs(self, feature : FeatureInfo):
		return feature.get_activs(self.cache[feature.input_layer.to_hookpoint_str()][0])

	@no_grad()
	def get_sae_activs_on_token(self, sae_info : SAEInfo, token_pos : int):
		activs = None
		inp = self.cache[sae_info.input_layer.to_hookpoint_str()][0, token_pos]
		try:
			sae_info.sae.to(device=device, dtype=dtype)
			activs = _to_numpy(sae_info.sae.get_activs(inp))
		finally:
			sae_info.sae.cpu()
		return activs

	@no_grad()
	def get_feature_ln_constant_at_token(self, feature : FeatureInfo, token_pos : int, use_unsteered : bool = False):
		if feature.input_layer.sublayer == Sublayer.ATTN_IN or (feature.input_layer.parallel_attn_mlp and feature.input_layer.sublayer == Sublayer.MLP_IN):
			pre_ln_layer = LayerSublayer(feature.input_layer.layer, Sublayer.RESID_PRE)
		elif feature.input_layer.sublayer == Sublayer.MLP_IN:
			pre_ln_layer = LayerSublayer(feature.input_layer.layer, Sublayer.RESID_MID)
		elif feature.input_layer.sublayer == Sublayer.PRE_LOGITS:
			pre_ln_layer = LayerSublayer(len(self.cache.model.blocks)-1, Sublayer.RESID_POST)
		else:
			return 1.0

		if use_unsteered:
			cache = self.unsteered_cache
		else:
			cache = self.cache

		pre_ln_activ = feature.get_activs(
			cache[pre_ln_layer.to_hookpoint_str()][0, token_pos],
			use_relu=False
		).item() # should be a single token
		post_ln_activ = feature.get_activs(cache[feature.input_layer.to_hookpoint_str()][0, token_pos], use_relu=False).item()

		if pre_ln_activ == 0: return 0
		return post_ln_activ/pre_ln_activ

	@no_grad()
	def get_features_activs_on_token(self, features : List[FeatureInfo], token_pos : int):
		retlist = []
		for feature in features:
			attrib = self.populate_attrib_info(AttribInfo(
				feature_info=feature,
				token_pos=token_pos
			))
			
			retlist.append(attrib)
		return retlist

	@no_grad()
	def _get_attn_head_contrib_tensors(self, model, layer, vector, dst_token_pos):
		split_vals = self.cache[
			f'blocks.{layer}.attn.hook_v'
		][0] # src head d_head
		# src head d_head, head d_head d_model -> src head d_model
		outs = torch.einsum('shf, hfm -> shm', split_vals, model.W_O[layer])
		# src head d_model, d_model -> head src
		value_contribs = torch.einsum('shm, m -> hs', outs, vector)

		patterns = self.cache[
			f'blocks.{layer}.attn.hook_pattern'
		][0] # head dst src
		patterns = patterns[:, dst_token_pos, :] # head src

		total_contribs = patterns * value_contribs # head src

		return patterns, value_contribs, total_contribs

	@no_grad()
	def get_top_attn_contribs(self, model : HookedTransformer, attrib : AttribInfo, k=7):
		feature = attrib.feature_info
		vector = feature.encoder_vector

		parallel_attn_mlp = model.cfg.parallel_attn_mlp
		max_layer = LayerSublayer(layer=feature.input_layer.layer, sublayer=feature.input_layer.sublayer, parallel_attn_mlp=parallel_attn_mlp)

		# get top contribs for each layer
		top_total_contribs_per_layer = []
		top_patterns_per_layer = []
		top_value_contribs_per_layer = []
		top_idxs_per_layer = []

		cur_layer = 0
		while LayerSublayer(layer=cur_layer, sublayer=Sublayer.ATTN_OUT, parallel_attn_mlp=parallel_attn_mlp) < max_layer:
			patterns, value_contribs, total_contribs = self._get_attn_head_contrib_tensors(model, cur_layer, vector, attrib.token_pos)
			top_attn_contribs, top_attn_idxs_flattened = torch.topk(
				total_contribs.flatten(),
				k=np.min(
					[k, np.prod(total_contribs.shape)]
				)
			)
			top_attn_idxs = torch.stack(
				torch.unravel_index(
					top_attn_idxs_flattened, total_contribs.shape
				)
			).T # [k, (head src)]

			top_patterns = patterns[top_attn_idxs[:,0], top_attn_idxs[:,1]]
			top_value_contribs = value_contribs[top_attn_idxs[:,0], top_attn_idxs[:,1]]

			top_total_contribs_per_layer.append(top_attn_contribs)
			top_patterns_per_layer.append(top_patterns)
			top_value_contribs_per_layer.append(top_value_contribs)
			top_idxs_per_layer.append(top_attn_idxs)

			cur_layer += 1

		# now, get top contribs total
		if len(top_total_contribs_per_layer) == 0:
			return [], []
		top_total_contribs_per_layer = torch.stack(top_total_contribs_per_layer)
		top_total_contribs_all, top_idxs_all = torch.topk(
			top_total_contribs_per_layer.flatten(),
			k=np.min(
				[k, np.prod(top_total_contribs_per_layer.shape)]
			)
		)
		top_idxs_all = torch.stack(
			torch.unravel_index(
				top_idxs_all, top_total_contribs_per_layer.shape
			)
		).T # [k, (layer, k_within_layer)]

		components = []
		contribs = []
		for layer_idx, layer_k in top_idxs_all:
			cur_head, cur_src_pos = top_idxs_per_layer[layer_idx][layer_k]
			cur_component = ComponentInfo(
				component_type=ComponentType.ATTN_HEAD,
				token_pos=cur_src_pos.item(),
				attn_head=cur_head.item(),
				attn_layer=layer_idx.item()
			)
			components.append(cur_component)
			contribs.append(
				top_total_contribs_per_layer[layer_idx, layer_k].item()
			)
		return components, contribs

	@no_grad()
	def get_top_sae_contribs(self, model : HookedTransformer, sae_dict : IdDict, attrib : AttribInfo, k=7, top_mlp_k:Optional[int]=None):
		with sae_dict.lock:
			feature = attrib.feature_info
			vector = feature.encoder_vector

			parallel_attn_mlp = model.cfg.parallel_attn_mlp
			max_layer = LayerSublayer(layer=feature.input_layer.layer, sublayer=feature.input_layer.sublayer, parallel_attn_mlp=parallel_attn_mlp)

			# if top_mlp_k is set, then only look at SAE features from the top_mlp_k MLP sublayers
			top_mlp_idxs = None
			if top_mlp_k is not None:
				mlp_contribs = []
				for layer in range(len(model.blocks)):
					mlp_out = self.cache[f'blocks.{layer}.hook_mlp_out'][0, attrib.token_pos]
					mlp_contribs.append(torch.dot(vector, mlp_out))
				top_mlp_idxs = torch.topk(torch.tensor(mlp_contribs), k=top_mlp_k)[1].to_list()

			# get top contribs for each SAE
			top_contribs_per_sae = []
			top_features_per_sae = []
			sae_idxs = []
			for sae_idx, sae_info in sae_dict.dict.items():
				cur_layer = LayerSublayer(layer=sae_info.output_layer.layer, sublayer=sae_info.output_layer.sublayer, parallel_attn_mlp=parallel_attn_mlp)
				if not (cur_layer < max_layer): continue
				if top_mlp_idxs is not None and cur_layer.layer not in top_mlp_idxs: continue

				sae_info.sae.to(device=device, dtype=dtype)
				activs = sae_info.sae.get_activs(
					self.cache[sae_info.input_layer.to_hookpoint_str()][0,attrib.token_pos]
				)
				pullback = (sae_info.sae.W_dec @ vector)
				total_contribs = activs * pullback
				sae_info.sae.cpu()

				top_contribs, top_features = torch.topk(total_contribs, k=k)

				top_contribs_per_sae.append(top_contribs)
				top_features_per_sae.append(top_features)
				sae_idxs.append(sae_idx)

			# now, get top contribs total
			if len(top_contribs_per_sae) == 0: return [], []
			top_contribs_per_sae = torch.stack(top_contribs_per_sae)
			top_contribs_all, top_idxs_all = torch.topk(
				top_contribs_per_sae.flatten(),
				k=np.min(
					[k, np.prod(top_contribs_per_sae.shape)]
				)
			)
			top_idxs_all = torch.stack(
				torch.unravel_index(
					top_idxs_all, top_contribs_per_sae.shape
				)
			).T

			components = []
			contribs = []
			for cur_idx, cur_k in top_idxs_all:
				cur_component = ComponentInfo(
					component_type=ComponentType.SAE_FEATURE,
					sae_idx=sae_idxs[cur_idx],
					feature_idx=top_features_per_sae[cur_idx][cur_k].item(),
					token_pos=attrib.token_pos,
				)
				components.append(cur_component)
				contribs.append(
					top_contribs_per_sae[cur_idx, cur_k].item()
				)
		return components, contribs

	@no_grad()
	def get_top_contribs(self, model : HookedTransformer, sae_dict : IdDict, attrib : AttribInfo, k=7, top_mlp_k=None):
		# print(f"Getting top contribs for node at layer {attrib.feature_info.input_layer}")
		attn_components, attn_contribs = self.get_top_attn_contribs(model, attrib, k=k)
		sae_components, sae_contribs = self.get_top_sae_contribs(model, sae_dict, attrib, k=k, top_mlp_k=top_mlp_k)

		embedding_component = ComponentInfo(
			component_type=ComponentType.EMBED,
			token_pos=attrib.token_pos,
			embed_vocab_idx=model.to_single_token(self.tokens[attrib.token_pos])
		)
		embedding_vec = model.W_E[embedding_component.embed_vocab_idx]
		embedding_contrib = torch.dot(embedding_vec, attrib.feature_info.encoder_vector)

		all_components = attn_components + sae_components + [embedding_component]
		all_contribs = attn_contribs + sae_contribs + [embedding_contrib]

		all_contribs_tensor = torch.tensor(all_contribs)
		all_contribs_tensor = all_contribs_tensor * attrib.total_ln_constant * attrib.total_invar_factor * attrib.total_attn_factor
		top_contribs, top_idxs = torch.topk(all_contribs_tensor, k=min(k, len(all_contribs_tensor)))

		top_components = [all_components[i.item()] for i in top_idxs]
		top_contribs = [all_contribs_tensor[i.item()].item() for i in top_idxs]

		return top_components, top_contribs
	
	def populate_attrib_info(self, new_attrib, old_attrib=None, use_unsteered=False):
		if use_unsteered:
			cache = self.unsteered_cache
		else:
			cache = self.cache
		
		# get attn factor (if applicable)
		attn_head = new_attrib.feature_info.attn_head
		if attn_head is not None and old_attrib is not None:
			new_attrib.attn_factor = cache[
				f'blocks.{new_attrib.feature_info.input_layer.layer}.attn.hook_pattern'
			][0, attn_head, old_attrib.token_pos, new_attrib.token_pos].item()

		# get feature activ
		new_attrib.feature_activ = new_attrib.feature_info.get_activs(
			cache[
				new_attrib.feature_info.input_layer.to_hookpoint_str()
			][0, new_attrib.token_pos]
		).item()

		# get the invariant factor
		if old_attrib is not None:
			new_attrib.invar_factor = torch.dot(new_attrib.feature_info.decoder_vector, old_attrib.feature_info.encoder_vector).item()
		else:
			new_attrib.invar_factor = 1.0

		# get the LN constant
		new_attrib.ln_constant = self.get_feature_ln_constant_at_token(new_attrib.feature_info, new_attrib.token_pos, use_unsteered=use_unsteered)

		# update total attribs if we have no parent old_attrib
		if old_attrib is None:
			new_attrib.total_ln_constant = new_attrib.ln_constant
			new_attrib.total_attn_factor = 1.0
			new_attrib.total_invar_factor = 1.0
			new_attrib.total_attrib = new_attrib.feature_activ
		else:
			new_attrib.total_ln_constant = old_attrib.total_ln_constant * new_attrib.ln_constant
			new_attrib.total_attn_factor = new_attrib.attn_factor * old_attrib.total_attn_factor
			new_attrib.total_invar_factor = new_attrib.invar_factor * old_attrib.total_invar_factor
			new_attrib.total_attrib = new_attrib.total_invar_factor * old_attrib.total_ln_constant * new_attrib.total_attn_factor * new_attrib.feature_activ

			new_attrib.parent_ln_constant = old_attrib.ln_constant
			new_attrib.total_parent_ln_constant = old_attrib.total_ln_constant

		# now, update new_attrib.unsteered_attrb
		if not use_unsteered: # prevent infinite recursion on child object
			if self.unsteered_cache is not None and new_attrib.unsteered_attrib is None:
				# prompt has been steered, but new_attrib doesn't reflect this
				new_attrib.unsteered_attrib = AttribInfo(
					feature_info=new_attrib.feature_info,
					token_pos=new_attrib.token_pos
				)
				new_attrib.unsteered_attrib = self.populate_attrib_info(new_attrib.unsteered_attrib, old_attrib.unsteered_attrib if old_attrib is not None else None, use_unsteered=True)
			elif self.unsteered_cache is None and new_attrib.unsteered_attrib is not None:
				# all steering vectors have been removed from the prompt, but new_attrib hasn't been updated to reflect this

				# TODO: make sure that there are no bugs here
				new_attrib.invar_factor = new_attrib.unsteered_attrib.invar_factor
				new_attrib.ln_constant = new_attrib.unsteered_attrib.ln_constant
				new_attrib.attn_factor = new_attrib.unsteered_attrib.attn_factor
				new_attrib.feature_activ = new_attrib.unsteered_attrib.feature_activ

				new_attrib.total_invar_factor = new_attrib.unsteered_attrib.total_invar_factor
				new_attrib.total_ln_constant = new_attrib.unsteered_attrib.total_ln_constant
				new_attrib.total_attn_factor = new_attrib.unsteered_attrib.total_attn_factor
				new_attrib.total_attrib = new_attrib.unsteered_attrib.total_attrib

				# TODO: how to deal with child components? and parent ln constant?

				new_attrib.unsteered_attrib = None

		return new_attrib

	@no_grad()
	def get_child_component_attrib_info(self, model : HookedTransformer, sae_dict : IdDict, attrib : AttribInfo, child : ComponentInfo, use_unsteered=False):
		new_feature = FeatureInfo()
		new_attrib = AttribInfo()
		new_attrib.token_pos = child.token_pos
		if child.component_type == ComponentType.SAE_FEATURE:
			with sae_dict.lock:
				sae_info = sae_dict.dict[child.sae_idx]
				feature_idx = child.feature_idx

				# make FeatureInfo
				new_feature = FeatureInfo.init_from_sae_feature(sae_info, feature_idx)
				new_feature.sae_info = sae_info
		elif child.component_type == ComponentType.ATTN_HEAD:
			assert(attrib is not None)
			# make FeatureInfo
			new_feature = FeatureInfo.init_from_attn_pullback(model, attrib.feature_info, child.attn_layer, child.attn_head)
		elif child.component_type == ComponentType.EMBED:
			embedding_vec = model.W_E[child.embed_vocab_idx]
			new_feature = FeatureInfo.init_from_vector(model, LayerSublayer(0, Sublayer.EMBED), embedding_vec, name=f'embed_{child.embed_vocab_idx}', contravariant=True)

		new_attrib.feature_info = new_feature

		# populate new_attrib with feature activ, invariant factor, LN constant
		new_attrib = self.populate_attrib_info(new_attrib, attrib, use_unsteered=use_unsteered)

		return new_attrib
	
	def add_steering_vector(self, model, steering_vector):
		idx = self.steering_vectors.add(steering_vector)
		self.run_with_steering_vectors(model)
		return idx
	
	def remove_steering_vector(self, model, steering_vector_id):
		self.steering_vectors.remove(steering_vector_id)
		if model is not None:
			self.run_with_steering_vectors(model)

	@no_grad()
	def generate_tokens(self, model, max_new_tokens=20, temperature=0.8, use_steering_vectors=True):
		hooks = []
		if use_steering_vectors and len(self.steering_vectors.dict) > 0:
			for steering_vector in self.steering_vectors.dict.values():
				hooks.extend(steering_vector.make_hooks())
		with model.hooks(fwd_hooks=hooks):
			generation = model.generate("".join(self.tokens), max_new_tokens=max_new_tokens, use_past_kv_cache=False)
		return generation

@dataclass
class ModelInfo:
	name : str
	path : str
	n_layers : int
	n_params : int

	dtype : Optional[torch.dtype] = None

	@classmethod
	def load_from_hooked_transformer(cls, model):
		return cls(
			name=model.cfg.model_name,
			path=model.cfg.model_name,
			n_layers=model.cfg.n_layers,
			n_params=model.cfg.n_params,
			dtype=model.dtype
		)

	def serialize(self):
		return {
			'name': self.name,
			'path': self.path,
			'n_layers': self.n_layers,
			'n_params': self.n_params,
			'dtype': str(self.dtype)
		}
	
	@classmethod
	def deserialize(cls, d):
		return cls(
			name=d['name'],
			path=d['path'],
			n_layers=d['n_layers'],
			n_params=d['n_params'],
			dtype=DTYPE_DICT[d.get('dtype', None)]
		)

class Session:
	def __init__(self):
		self.name = None
		self.description = None

		self.model_info = None
		self.model = None

		self.metadata_only = False

		self.prompt_list = IdDict() # each element will be a Prompt
		self.sae_list = IdDict() # each element will be an SAEInfo
		self.all_feature_lists = IdDict() # each element will be another IdDict, which in turn will contain a list of features
	
	def save_feature_list(self, feature_list, dirpath, json_filename="feature_list.json", save_tensors=True, out_zipfile=None):
		if out_zipfile is None:
			if not os.path.exists(dirpath):
				os.mkdir(dirpath)
		"""else:
			if not zipfile.Path(out_zipfile, dirpath).exists():
				out_zipfile.mkdir(dirpath)"""

		retdict = {
			'name': feature_list.name,
			'description': feature_list.description,
			'feature_json_filenames': {}
		}
		with feature_list.lock:
			for idx, feature in feature_list.dict.items():
				feature_dirname = f'feature_{idx}'
				feature_json_filename = feature.save(os.path.join(dirpath, feature_dirname), save_tensors=save_tensors, out_zipfile=out_zipfile)
				retdict['feature_json_filenames'][idx] = os.path.join(feature_dirname, feature_json_filename)

		if out_zipfile is None:
			with open(os.path.join(dirpath, json_filename), "w") as ofp:
				json.dump(retdict, ofp)
		else:
			with out_zipfile.open(os.path.join(dirpath, json_filename), "w") as ofp:
				s = json.dumps(retdict)
				ofp.write(bytes(s, 'ascii'))

		return json_filename
	
	def load_feature_list(self, json_path, load_tensors=True, in_zipfile=None):
		if in_zipfile is None:
			with open(json_path, "r") as fp:
				d = json.load(fp)
		else:
			with in_zipfile.open(json_path, "r") as fp:
				d = json.load(fp)

		feature_list = IdDict()
		feature_list.name = d['name']
		feature_list.description = d['description']

		dirname = os.path.dirname(json_path)
		for idx, feature_json_filename in d['feature_json_filenames'].items():
			feature = FeatureInfo.load(os.path.join(dirname, feature_json_filename), load_tensors=load_tensors, in_zipfile=in_zipfile)
			feature_list.add(feature)

		return feature_list
	
	def save_wrapper(self, path, json_filename="session.json", sae_metadata_only=True, save_feature_tensors=True):
		if (ext := os.path.splitext(path)[1]) == '.peek' or ext == '.zip':
			with zipfile.ZipFile(path, "w") as out_zipfile:
				self.save('', json_filename=json_filename, sae_metadata_only=sae_metadata_only, save_feature_tensors=save_feature_tensors, out_zipfile=out_zipfile)
		else:
			self.save(path, json_filename=json_filename, sae_metadata_only=sae_metadata_only, save_feature_tensors=save_feature_tensors, out_zipfile=None)
	def save(self, dirpath, json_filename="session.json", sae_metadata_only=True, save_feature_tensors=True, out_zipfile=None):
		if out_zipfile is None and not os.path.exists(dirpath):
			os.mkdir(dirpath)

		retdict = {
			'name': self.name,
			'description': self.description,
			'model_info': self.model_info.serialize(),

			'metadata_only': not save_feature_tensors,

			'prompt_json_filenames': {},
			'feature_list_json_filenames': {},
			'sae_info_json_filenames': {},
		}

		with self.prompt_list.lock:
			for idx, prompt in self.prompt_list.dict.items():
				prompt_dirname = f'prompt_{idx}'
				prompt_filename = prompt.save(os.path.join(dirpath, prompt_dirname), save_tensors=save_feature_tensors, out_zipfile=out_zipfile)
				retdict['prompt_json_filenames'][idx] = os.path.join(prompt_dirname, prompt_filename)
		
		with self.all_feature_lists.lock:
			for idx, feature_list in self.all_feature_lists.dict.items():
				feature_list_dirname = f'feature_list_{idx}'
				feature_list_json_filename = self.save_feature_list(feature_list, os.path.join(dirpath, feature_list_dirname), save_tensors=save_feature_tensors, out_zipfile=out_zipfile)
				retdict['feature_list_json_filenames'][idx] = os.path.join(feature_list_dirname, feature_list_json_filename)

		with self.sae_list.lock:
			for idx, sae_info in self.sae_list.dict.items():
				sae_dirname = f'sae_{idx}'
				sae_info_json_filename = sae_info.save(os.path.join(dirpath, sae_dirname), save_tensors=(not sae_metadata_only), out_zipfile=out_zipfile)
				retdict['sae_info_json_filenames'][idx] = os.path.join(sae_dirname, sae_info_json_filename)

		if out_zipfile is None:
			with open(os.path.join(dirpath, json_filename), "w") as ofp:
				json.dump(retdict, ofp)
		else:
			with out_zipfile.open(os.path.join(dirpath, json_filename), "w") as ofp:
				s = json.dumps(retdict)
				ofp.write(bytes(s, 'ascii'))

		return json_filename

	def load_wrapper(self, path, load_tensors=True):
		if not os.path.exists(path):
			raise Exception(f"Error: no file found at {path}.")

		if zipfile.is_zipfile(path):
			with zipfile.ZipFile(path, 'r') as in_zipfile:
				self.load('session.json', load_tensors=load_tensors, in_zipfile=in_zipfile)
		else:
			json_path = path
			old_json_path = json_path
			if os.path.isdir(json_path):
				json_path = os.path.join(json_path, "session.json")
			if not os.path.exists(json_path):
				raise Exception(f"Error: no session json file found in {old_json_path}")
			self.load(json_path, load_tensors=load_tensors, in_zipfile=None)

	def load(self, json_path, load_tensors=True, in_zipfile=None):
		if in_zipfile is None:
			with open(json_path, "r") as fp:
				d = json.load(fp)
		else:
			with in_zipfile.open(json_path, "r") as fp:
				d = json.load(fp)

		self.name = d['name']
		self.description = d['description']

		if load_tensors:
			self.load_model_from_pretrained(d['model_info']['path'])
		else:
			self.model_info = ModelInfo.deserialize(d['model_info'])

		dirname = os.path.dirname(json_path)

		self.sae_list = IdDict()
		self.prompt_list = IdDict()
		self.all_feature_lists = IdDict()

		with self.sae_list.lock:
			# need to directly update the sae_list dict to ensure that all indices match
			self.sae_list.dict = {}
			self.sae_list.cur_id = 0
			for idx_str, sae_info_json_filename in d['sae_info_json_filenames'].items():
				idx = int(idx_str)
				sae_info = SAEInfo.load(os.path.join(dirname, sae_info_json_filename), load_tensors=False, in_zipfile=in_zipfile)
				if load_tensors: sae_info = sae_info.load_from_upstreams(load_tensors=load_tensors)

				self.sae_list.dict[idx] = sae_info
				if idx >= self.sae_list.cur_id:
					self.sae_list.cur_id = idx + 1

		with self.all_feature_lists.lock:
			# need to directly update the feature list dict to ensure that all indices match
			self.all_feature_lists.dict = {}
			self.all_feature_lists.cur_id = 0
			for idx_str, feature_list_json_filename in d['feature_list_json_filenames'].items():
				idx = int(idx_str)
				feature_list = self.load_feature_list(os.path.join(dirname, feature_list_json_filename), load_tensors=load_tensors, in_zipfile=in_zipfile)
				self.all_feature_lists.dict[idx] = feature_list
				if idx >= self.all_feature_lists.cur_id:
					self.all_feature_lists.cur_id = idx + 1

		for idx, prompt_json_filename in d['prompt_json_filenames'].items():
			prompt = Prompt.load(os.path.join(dirname, prompt_json_filename), load_tensors=load_tensors, in_zipfile=in_zipfile)
			if load_tensors:
				# fill up prompt cache
				# start from token 1 so that we don't re-introduce the bos token
				prompt.run_model_on_text(self.model, "".join(prompt.tokens[1:]))
			self.prompt_list.add(prompt)
	
	def render(self, outpath="output.html"):
		render.render_sess(self, outpath=outpath)

	# most Session methods are intended to be easily exposed as HTTP APIs
	# hence, you might see some methods here that one might expect to find belonging to other classes

	# model methods

	def load_model_from_pretrained(self, path, name=None, dtype_str=None):
		self.model = HookedTransformer.from_pretrained(
			path, 
			dtype=DTYPE_DICT[dtype_str],
			device=device
		)
		if name is None: name = path
		self.model_info = ModelInfo(
			name=name,
			path=path,
			n_layers=self.model.cfg.n_layers,
			n_params=self.model.cfg.n_params,
			dtype=self.model.W_E.dtype
		)

		global dtype
		dtype = self.model.W_E.dtype

	def get_model_info(self):
		return self.model_info.serialize()

	# PromptList methods

	def _get_prompt_info_dict(self, prompt):
		retval = {}
		retval['name'] = prompt.name
		retval['description'] = prompt.description
		retval['tokens'] = prompt.tokens
		retval['cur_feature_list_idx'] = prompt.cur_feature_list_idx
		return retval

	def list_info_for_all_prompts(self):
		retlist = []
		with self.prompt_list.lock:
			for idx, prompt in self.prompt_list.dict.items():
				retval = self._get_prompt_info_dict(prompt)
				retval['id'] = idx
				retlist.append(retval)
		return retlist
	
	def list_info_for_prompt_by_id(self, idx):
		retdict = self._get_prompt_info_dict(self.prompt_list.dict[idx])
		retdict['id'] = idx
		return retdict

	def create_new_prompt(self, name=None, description=None):
		if name is None: name = f'Prompt {self.prompt_list.cur_id}'
		prompt = Prompt(name=name, description=description)
		# set current feature list to be the first one in our dictionary
		if len(self.all_feature_lists.dict) > 0:
			prompt.cur_feature_list_idx = next(iter(self.all_feature_lists.dict))
		return self.prompt_list.add(prompt)

	def delete_prompt_by_id(self, idx):
		self.prompt_list.remove(idx)

	# SAEList methods
	def list_info_for_all_saes(self):
		retlist = []
		with self.sae_list.lock:
			for idx, sae_info in sorted(self.sae_list.dict.items(), key=lambda x: x[1].input_layer):
				retval = {**sae_info.serialize(), **sae_info.sae.cfg.serialize()}
				retval['id'] = idx

				retlist.append(retval)
		return retlist

	def delete_sae_by_id(self, idx):
		self.sae_list.remove(idx)

	def load_saes_from_hf_repo(self, hf_repo_id, sae_info_json_filename="sae_info.json"):
		retlist = []
		local_path = hf_hub.snapshot_download(hf_repo_id)
		for name in os.listdir(local_path):
			json_path = os.path.join(name, sae_info_json_filename)
			if os.path.exists(os.path.join(local_path, json_path)):
				sae_info = SAEInfo.load_from_hf(hf_repo_id, json_path, load_tensors=True)
				retlist.append(self.sae_list.add(sae_info))
		return retlist

	def load_sae_from_path(self, path, sae_info_json_filename="sae_info.json"):
		if not os.path.exists(path):
			raise Exception(f"File not found: \"{path}\"")
		if os.path.isdir(path):
			retlist = []
			for filename in os.listdir(path):
				fullpath = os.path.join(path, filename, sae_info_json_filename)
				try:
					sae_info = SAEInfo.load(fullpath, load_tensors=False)
					sae_info = sae_info.load_from_upstreams(load_tensors=True)
					retlist.append(self.sae_list.add(sae_info))
				except Exception as e:
					 print(e)
			return retlist
		else:
			try:
				sae_info = SAEInfo.load(fullpath, load_tensors=False)
				sae_info = sae_info.load_from_upstreams(load_tensors=True)
				return self.sae_list.add(sae_info)
			except:
				raise Exception(f"Error loading SAE from \"{path}\". Perhaps this is an invalid file?")

	def rename_sae(self, idx, name):
		sae_info = sae_list.dict[idx]
		sae_info.short_name = name

	# List of FeatureLists - related methods
	def list_info_for_all_feature_lists(self):
		retlist = []
		with self.all_feature_lists.lock:
			for idx, cur_feature_list in self.all_feature_lists.dict.items():
				retval = {}
				retval['id'] = idx
				retval['name'] = cur_feature_list.name
				retval['description'] = cur_feature_list.description
				retval['num_features'] = len(cur_feature_list.dict)

				retlist.append(retval)
		return retlist

	def delete_feature_list_by_id(self, idx):
		# we need to re-assign feature lists to all the prompts that were using this feature list
		# to do this, we'll have to figure out which feature list these prompts should get
		if len(self.all_feature_lists.dict) <= 1:
			raise Exception("Can't delete the last feature list.")
		with self.all_feature_lists.lock:
			idxs = list(self.all_feature_lists.dict.keys())
			idx_index = idxs.index(idx)
			if idx_index != 0:
				new_idx = idxs[idx_index-1]
			else:
				new_idx = idxs[1]

			with self.prompt_list.lock:
				for prompt in self.prompt_list.dict.values():
					if prompt.cur_feature_list_idx == idx:
						prompt.cur_feature_list_idx = new_idx

		# finally, we can remove the feature list
		self.all_feature_lists.remove(idx)

	def create_feature_list(self, name=None, description=None, copy_from_id=None):
		new_feature_list = IdDict()
		if copy_from_id is not None:
			old_feature_list = self.all_feature_lists.dict[copy_from_id]
			with old_feature_list.lock:
				for idx, feature in old_feature_list.dict.items():
					new_feature_list.add(feature)
		if name is not None:
			new_feature_list.name = name
		else:
			if copy_from_id is not None:
				new_feature_list.name = old_feature_list.name + " (Copy)"
			else:
				new_feature_list.name = f"Untitled{self.all_feature_lists.cur_id}"
		if description is not None:
			new_feature_list.description = description
		else:
			if copy_from_id is not None:
				new_feature_list.description = old_feature_list.description
		return self.all_feature_lists.add(new_feature_list)

	def rename_feature_list(self, idx, name=None, description=None):
		with self.all_feature_lists.lock:
			cur_feature_list = self.all_feature_lists.dict[idx]
			if name is not None: cur_feature_list.name = name
			if description is not None: cur_feature_list.description = description

	# Prompt-related methods
	def rename_prompt(self, idx, name=None, description=None):
		with self.prompt_list.lock:
			prompt = self.prompt_list.dict[idx]
			if name is not None: prompt.name = name
			if description is not None: prompt.description = description

	def run_model_on_prompt_text(self, idx, text):
		return self.prompt_list.dict[idx].run_model_on_text(self.model, text)

	# FeatureList-related methods
	def get_cur_feature_list_idx_for_prompt(self, idx):
		return self.prompt_list.dict[idx].cur_feature_list_idx

	def change_cur_feature_list_for_prompt(self, prompt_idx, feature_list_idx):
		self.prompt_list.dict[prompt_idx].cur_feature_list_idx = feature_list_idx

	def _get_feature_info_dict(self, feature):
		"""cur_retval = {}
		cur_retval['name'] = feature.name
		cur_retval['description'] = feature.description

		cur_retval['input_layer'] = feature.input_layer.layer
		cur_retval['input_sublayer'] = feature.input_layer.sublayer.name
		cur_retval['output_layer'] = feature.output_layer.layer
		cur_retval['output_sublayer'] = feature.output_layer.sublayer.name

		cur_retval['feature_type'] = feature.feature_type.name
		cur_retval['feature_idx'] = feature.feature_idx
		cur_retval['attn_head'] = feature.attn_head
		cur_retval['observable_tokens'] = feature.observable_tokens
		cur_retval['observable_weights'] = feature.observable_weights

		return cur_retval"""
		return feature.serialize()

	def get_feature_info_dict_by_idx(self, feature_list_idx, feature_idx):
		feature = self.all_feature_lists.dict[feature_list_idx].dict[feature_idx]

		return feature.serialize() #self._get_feature_info_dict(feature)

	def get_feature_list_info(self, feature_list_idx):
		feature_list = self.all_feature_lists.dict[feature_list_idx]

		retdict = {}
		retdict['features'] = []
		retdict['name'] = feature_list.name
		retdict['description'] = feature_list.description
		retdict['id'] = feature_list_idx

		with feature_list.lock:
			for feature_idx, feature in feature_list.dict.items():
				cur_retval = self.get_feature_info_dict_by_idx(feature_list_idx, feature_idx)
				cur_retval['id'] = feature_idx
				retdict['features'].append(cur_retval)

		return retdict

	def add_feature_from_observable(self, feature_list_idx, observable_tokens, observable_weights, name=None, description=None, do_unembed_pullback=False):
		feature = FeatureInfo.init_from_observable(self.model, observable_tokens, observable_weights, name=name, description=description, do_unembed_pullback=do_unembed_pullback)
		new_id = self.all_feature_lists.dict[feature_list_idx].add(feature)
		return new_id

	def add_feature_from_sae(self, feature_list_idx, sae_idx, sae_feature_idx, name=None, description=None):
		sae_info = self.sae_list.dict[sae_idx]
		feature = FeatureInfo.init_from_sae_feature(sae_info, sae_feature_idx, name=name, description=description)
		feature.sae_idx = sae_idx
		return self.all_feature_lists.dict[feature_list_idx].add(feature)

	def remove_feature_from_list(self, feature_list_idx, feature_idx):
		self.all_feature_lists.dict[feature_list_idx].remove(feature_idx)

	def rename_feature(self, feature_list_idx, feature_idx, name=None, description=None):
		feature = self.all_feature_lists.dict[feature_list_idx].dict[feature_idx]
		if name is not None:
			feature.name = name
		if description is not None:
			feature.description = description

	def get_feature_activs_on_prompt(self, prompt_idx, feature_list_idx, feature_idx):
		feature = self.all_feature_lists.dict[feature_list_idx].dict[feature_idx]
		prompt = self.prompt_list.dict[prompt_idx]
		return [x.item() for x in prompt.get_feature_activs(feature)]

	def get_feature_list_activs_on_token(self, prompt_idx, token_pos, feature_list_idx):
		retlist = []
		prompt = self.prompt_list.dict[prompt_idx]
		feature_list = self.all_feature_lists.dict[feature_list_idx]
		with feature_list.lock:
			for idx, feature in feature_list.dict.items():
				# deal with metadata-only features
				if feature.encoder_vector is None: continue

				cur_dict = {}
				cur_dict['activation'] = feature.get_activs(
					prompt.cache[feature.input_layer.to_hookpoint_str()][0, token_pos]
				).item()
				
				# deal with steering
				if prompt.unsteered_cache is not None:
					cur_dict['unsteered_activation'] = feature.get_activs(
						prompt.unsteered_cache[feature.input_layer.to_hookpoint_str()][0, token_pos]
					).item()
				else:
					cur_dict['unsteered_activation'] = None

				cur_dict['name'] = feature.name
				cur_dict['id'] = idx

				retlist.append(cur_dict)
		return retlist

	def get_sae_activs_on_token(self, prompt_idx, token_pos, sae_idx, k=None):
		sae_info = self.sae_list.dict[sae_idx]
		activs = self.prompt_list.dict[prompt_idx].get_sae_activs_on_token(sae_info, token_pos)
		if k is not None:
			with no_grad():
				top_activs, top_features = torch.topk(torch.from_numpy(activs), k=k)
			return {'scores': [x.item() for x in top_activs], 'feature_idxs': [x.item() for x in top_features]}
		else:
			return {'scores': [float(x) for x in activs], 'feature_idxs': None}

	# ComputationalPath-related methods
	def list_comp_paths_for_prompt(self, prompt_idx):
		retlist = []

		prompt = self.prompt_list.dict[prompt_idx]
		with prompt.comp_paths.lock:
			for idx, path in prompt.comp_paths.dict.items():
				retdict = {}
				retdict['name'] = path.name
				retdict['description'] = path.description

				nodes = []
				for node in path.get_total_attribs():
					nodedict = {}
					nodedict['token_pos'] = node.token_pos
					nodedict['total_attrib'] = node.total_attrib
					nodedict['description'] = node.description
					nodedict['name'] = node.name if node.name is not None else node.feature_info.name
					nodes.append(nodedict)

				retdict['nodes'] = nodes
				retdict['id'] = idx

				retdict['is_outdated'] = path.is_outdated
				retdict['outdated_token_strs'] = path.outdated_token_strs

				retlist.append(retdict)

		return retlist

	def delete_comp_path_by_id(self, prompt_idx, path_idx):
		self.prompt_list.dict[prompt_idx].comp_paths.remove(path_idx)
	
	def set_cur_comp_path_to_feature(self, prompt_idx, token_pos, feature_idx):
		prompt = self.prompt_list.dict[prompt_idx]
		feature_list = self.all_feature_lists.dict[prompt.cur_feature_list_idx]
		feature_info = feature_list.dict[feature_idx]
		new_attrib = prompt.get_features_activs_on_token([feature_info], token_pos)[0]
		new_comp_path = ComputationalPath(name="Current computational path", description="")
		new_comp_path.nodes = [new_attrib]
		prompt.cur_comp_path = new_comp_path

	def select_and_view_comp_path(self, prompt_idx, path_idx=None, feature_pos=-1, top_k_children=7, top_mlp_k=None):
		retdict = {}

		# get computational path
		prompt = self.prompt_list.dict[prompt_idx]
		if path_idx is None:
			comp_path = prompt.cur_comp_path
		else:
			comp_path = prompt.comp_paths.dict[path_idx]

		if len(comp_path.nodes) == 0:
			return {'nodes': [], 'top_children': []}

		# include information about whether the path is outdated
		retdict['is_outdated'] = comp_path.is_outdated
		retdict['outdated_token_strs'] = comp_path.outdated_token_strs

		# get attribution info for each node
		retdict['nodes'] = []
		all_attribs = comp_path.get_total_attribs()
		for node in all_attribs:
			nodedict = copy.copy(self._get_feature_info_dict(node.feature_info))
			nodedict = {**nodedict, **node.serialize_base()}
			if node.unsteered_attrib is not None:
				nodedict['unsteered_attrib'] = node.unsteered_attrib.serialize_base()
			else:
				nodedict['unsteered_attrib'] = None

			nodedict['name'] = node.name if node.name is not None else node.feature_info.name
			nodedict['description'] = node.description

			# get top-contributing child components
			if not comp_path.is_outdated:
				if node.top_child_components is None:
					# print(f"About to get top contribs for node at layer {node.feature_info.input_layer}")
					top_components, top_contribs = prompt.get_top_contribs(self.model, self.sae_list, node, k=top_k_children, top_mlp_k=top_mlp_k)
					node.top_child_components = top_components
					node.top_child_contribs = top_contribs
				nodedict['top_children'] = []
				for component, contrib in zip(node.top_child_components, node.top_child_contribs):
					childdict = {}
					childdict['component_type'] = component.component_type.name
					childdict['token_pos'] = component.token_pos

					childdict['attn_head'] = component.attn_head
					childdict['attn_layer'] = component.attn_layer
					childdict['sae_idx'] = component.sae_idx
					childdict['feature_idx'] = component.feature_idx
					childdict['embed_vocab_idx'] = component.embed_vocab_idx

					childdict['contrib'] = contrib
					# deal with steering: also want to get child components' unsteered attribs
					if prompt.unsteered_cache is not None:
						# TODO: is this way too inefficient?
						unsteered_contrib = prompt.get_child_component_attrib_info(self.model, self.sae_list, node, component, use_unsteered=True).total_attrib
						childdict['unsteered_contrib'] = unsteered_contrib
					else:
						childdict['unsteered_contrib'] = None
						
					nodedict['top_children'].append(childdict)

			retdict['nodes'].append(nodedict)

		if not comp_path.is_outdated:
			retdict['top_children'] = retdict['nodes'][feature_pos]['top_children']
		else:
			retdict['top_children'] = []

		# set prompt's current computational path to the new one
		prompt.cur_comp_path = copy.copy(comp_path)

		# also pass back the feature_pos of the current component
		retdict['cur_node_idx'] = feature_pos
		return retdict
	
	def update_comp_path_node(self, prompt_idx, node_idx, name, description, path_idx=None):
		prompt = self.prompt_list.dict[prompt_idx]
		if path_idx is None:
			comp_path = prompt.cur_comp_path
		else:
			comp_path = prompt.comp_paths.dict[path_idx]
		comp_path.nodes[node_idx].name = name
		comp_path.nodes[node_idx].description = description

	def get_child_attrib_for_comp_path(self, prompt_idx, childdict, path_idx=None, top_k_children=7, extend=True, cur_node_idx=-1):
		prompt = self.prompt_list.dict[prompt_idx]

		child_component = ComponentInfo(getattr(ComponentType, childdict['component_type']), childdict['token_pos'])

		# make component from dict
		# TODO: there's gotta be a better way to do all this stuff
		# TODO error checking / validation
		attribute_names = ['attn_head','attn_layer','sae_idx','feature_idx','embed_vocab_idx']
		for attribute_name in attribute_names:
			try: val = childdict[attribute_name]
			except KeyError: val = None 
			setattr(child_component, attribute_name, val)

		# get attrib for child component
		prompt = self.prompt_list.dict[prompt_idx]
		if not extend or len(prompt.cur_comp_path.nodes) == 0:
			cur_attrib = None
		else:
			cur_attrib = prompt.cur_comp_path.nodes[cur_node_idx]
		new_attrib = prompt.get_child_component_attrib_info(self.model, self.sae_list, cur_attrib, child_component)

		if extend:
			# append resulting attrib object to current computational path
			# TODO: there should be a lot more locks everywhere

			#prompt.cur_comp_path.nodes.append(new_attrib)
			if cur_node_idx < 0:
				cur_node_idx = len(prompt.cur_comp_path.nodes) + cur_node_idx
			prompt.cur_comp_path.nodes = prompt.cur_comp_path.nodes[:cur_node_idx+1] + [new_attrib]
		else:
			prompt.cur_comp_path = ComputationalPath("")
			prompt.cur_comp_path.nodes = [new_attrib]

		prompt.cur_comp_path.name = ""
		prompt.cur_comp_path.description = ""

		# print(prompt.cur_comp_path.is_outdated)

		# now return information for the updated current computational path

		return self.select_and_view_comp_path(prompt_idx, top_k_children=top_k_children)

	def save_cur_comp_path(self, prompt_idx, name=None, description=None):
		prompt = self.prompt_list.dict[prompt_idx]
		new_comp_path = copy.deepcopy(prompt.cur_comp_path)

		if name is not None and name != "": new_comp_path.name = name
		else: new_comp_path.name = f'Untitled {prompt.comp_paths.cur_id}'

		new_comp_path.description = description

		return prompt.comp_paths.add(new_comp_path)

	def update_comp_path(self, prompt_idx, path_idx, name=None, description=None):
		comp_path = self.prompt_list.dict[prompt_idx].comp_paths.dict[path_idx]
		if name is not None: comp_path.name = name
		if description is not None: comp_path.description = description

	def remove_comp_path(self, prompt_idx, path_idx):
		self.prompt_list.dict[prompt_idx].comp_paths.remove(path_idx)

	def get_feature_info_from_comp_path(self, prompt_idx, path_idx=None, feature_pos=-1):
		if path_idx is None:
			comp_path = self.prompt_list.dict[prompt_idx].cur_comp_path
		else:
			comp_path = self.prompt_list.dict[prompt_idx].comp_paths.dict[path_idx]
		feature = comp_path.nodes[feature_pos].feature_info
		return self._get_feature_info_dict(feature)

	def add_feature_from_comp_path_to_feature_list(self, prompt_idx, path_idx, feature_list_idx, feature_pos=-1, name=None, description=None):
		if path_idx is None:
			comp_path = self.prompt_list.dict[prompt_idx].cur_comp_path
		else:
			comp_path = self.prompt_list.dict[prompt_idx].comp_paths.dict[path_idx]
		if feature_list_idx is None:
			feature_list_idx = self.prompt_list.dict[prompt_idx].cur_feature_list_idx
		feature = comp_path.nodes[feature_pos].feature_info
		feature = copy.copy(feature)
		if name is not None: feature.name = name
		if description is not None: feature.description = description
		return self.all_feature_lists.dict[feature_list_idx].add(feature)

	# misc input-invariant stuff
	def _top_k_input_invariant_features(self, sae_info, feature_info, k=7):
		input_invar_scores = sae_info.get_feature_input_invariant_scores(feature_info)
		top_scores, top_features = torch.topk(torch.from_numpy(input_invar_scores), k=k)
		return {'scores': [x.item() for x in top_scores], 'feature_idxs': [x.item() for x in top_features]}
	def top_input_invar_features_from_comp_path(self, prompt_idx, path_idx, sae_idx, feature_pos=-1, k=7):
		if path_idx is None:
			comp_path = self.prompt_list.dict[prompt_idx].cur_comp_path
		else:
			comp_path = self.prompt_list.dict[prompt_idx].comp_paths.dict[path_idx]
		feature = comp_path.nodes[feature_pos].feature_info
		sae_info = self.sae_list.dict[sae_idx]
		return self._top_k_input_invariant_features(sae_info, feature, k=k)
	def top_input_invar_features_from_feature_list(self, feature_list_idx, feature_idx, sae_idx, k=7):
		feature = self.all_feature_lists.dict[feature_list_idx].dict[feature_idx]
		sae_info = self.sae_list.dict[sae_idx]
		return self._top_k_input_invariant_features(sae_info, feature, k=k)

	def _top_k_deembeddings(self, feature_info, k=7):
		deembedding_scores = feature_info.get_deembeddings(self.model)
		top_scores, top_features = torch.topk(torch.from_numpy(deembedding_scores), k=k)
		return {'scores': [x.item() for x in top_scores], 'tokens': [self.model.to_single_str_token(x.item()) for x in top_features]}
	def top_deembeddings_from_comp_path(self, prompt_idx, path_idx, feature_pos=-1, k=7):
		if path_idx is None:
			comp_path = self.prompt_list.dict[prompt_idx].cur_comp_path
		else:
			comp_path = self.prompt_list.dict[prompt_idx].comp_paths.dict[path_idx]
		feature = comp_path.nodes[feature_pos].feature_info
		return self._top_k_deembeddings(feature, k=k)
	def top_deembeddings_from_feature_list(self, feature_list_idx, feature_idx, sae_idx, feature_pos=-1, k=7):
		feature = self.all_feature_lists.dict[feature_list_idx].dict[feature_idx]
		return self._top_k_deembeddings(feature, k=k)	
	
	# SteeringVector-related methods
	
	def add_steering_vector_from_comp_path_to_prompt(self, prompt_idx, comp_path_idx, steering_coefficient, feature_pos=-1, do_clamp=True, use_encoder=False, name=None, description=None, all_tokens=False):
		prompt = self.prompt_list.dict[prompt_idx]
		if comp_path_idx is None:
			comp_path = prompt.cur_comp_path
		else:
			comp_path = prompt.comp_paths.dict[comp_path_idx]

		attrib = comp_path.nodes[feature_pos]
		feature = attrib.feature_info

		steering_vector = SteeringVector(
			feature_info=feature,
			token_pos=attrib.token_pos if not all_tokens else None,
			coefficient=steering_coefficient,
			do_clamp=do_clamp,
			use_encoder=use_encoder,
			name=name if name is not None else feature.name,
			description=description if description is not None else feature.description
		)

		return prompt.add_steering_vector(self.model, steering_vector)
	
	def remove_steering_vector(self, prompt_idx, steering_vector_idx):
		self.prompt_list.dict[prompt_idx].remove_steering_vector(self.model, steering_vector_idx)
	
	def list_steering_vectors_for_prompt(self, prompt_idx):
		retlist = []
		prompt = self.prompt_list.dict[prompt_idx]
		with prompt.steering_vectors.lock:
			for idx, steering_vector in prompt.steering_vectors.dict.items():
				retdict = {
					'id': idx,
					'name': steering_vector.name,
					'description': steering_vector.description,

					'feature_info': steering_vector.feature_info.serialize(),
					'token_pos': steering_vector.token_pos,
					'coefficient': steering_vector.coefficient,
					'do_clamp': steering_vector.do_clamp
				}
				retlist.append(retdict)

		return retlist
