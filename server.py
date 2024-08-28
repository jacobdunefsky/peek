import json

import tc_peek_backend as backend

from flask import request, Flask, render_template

sess = backend.Session()
sess.name = 'Untitled'

# create a default feature list
default_feature_list_idx = sess.create_feature_list(name="Default feature list", description=None)
# create a default prompt
first_prompt_idx = sess.create_new_prompt(name="My first prompt", description=None)
sess.change_cur_feature_list_for_prompt(first_prompt_idx, default_feature_list_idx)

app = Flask(__name__, template_folder='.')

@app.get("/style.css")
def stylefile():
	with open("./style.css", "r") as fp:
		return fp.read()

@app.get('/')
def index():
	return render_template('./index.html')

@app.put('/session/render')
def render():
	try:
		outpath = request.json['path']
	except KeyError:
		outpath = 'output.html'
	
	try:
		sess.render(outpath=outpath)
	except Exception as e:
		return {"error": str(e)}, 500
	
	return {}, 204

# set up routes for all the Session methods
@app.post("/model")
def load_model():
	try:
		name = request.json['name']
	except KeyError:
		name = None
	
	try:
		sess.load_model_from_pretrained(name)
	except ValueError:
		# TODO: use standardized error codes instead of strings
		return {'error': 'Invalid model name'}, 400

	return {}, 204

@app.get('/model')
def get_model_info():
	return sess.get_model_info()

@app.get('/prompts')
def list_info_for_all_prompts():
	return sess.list_info_for_all_prompts()

@app.get('/prompts/<int:idx>')
def list_info_for_prompt(idx):
	return sess.list_info_for_prompt_by_id(idx)

@app.delete('/prompts/<int:idx>')
def delete_prompt(idx):
	sess.delete_prompt_by_id(idx)
	return {}, 204

@app.post('/prompts')
def create_new_prompt():
	name = request.json.get('name', None)
	description = request.json.get('description', None)
	new_id = sess.create_new_prompt(name=name, description=description)
	return {'prompt_id': new_id}

@app.get('/saes')
def list_info_for_all_saes():
	return sess.list_info_for_all_saes()

@app.delete('/saes/<int:idx>')
def delete_sae_by_id(idx):
	sess.delete_sae_by_id(idx)
	return {}, 204

@app.post('/saes')
def load_sae_from_path():
	try:
		if 'hf_repo' in request.json:
			hf_repo = request.json['hf_repo']
			new_id = sess.load_saes_from_hf_repo(hf_repo)
		elif 'path' in request.json:
			path = request.json['path']
			new_id = sess.load_sae_from_path(path)
	except Exception as e:
		print(traceback.format_exc())
		return {'error': str(e)}, 500
	return {'sae_id': new_id}

@app.put('/saes/<int:idx>')
def rename_sae(idx):
	name = request.json['name']
	sess.rename_sae(idx, name)
	return {}, 204

@app.get('/feature_lists')
def list_info_for_all_feature_lists():
	return sess.list_info_for_all_feature_lists()

@app.delete('/feature_lists/<int:idx>')
def delete_feature_list_by_id(idx):
	sess.delete_feature_list_by_id(idx)
	return {}, 204

@app.post('/feature_lists')
def create_feature_list():
	name = request.json.get('name', None)
	description = request.json.get('description', None)
	copy_from_id = request.json.get('copy_from_id', None)
	if copy_from_id is not None: copy_from_id = int(copy_from_id)
	new_id = sess.create_feature_list(name=name, description=description, copy_from_id=copy_from_id)
	return {'feature_list_id': new_id}

@app.put('/feature_lists/<int:idx>')
def rename_feature_list(idx):
	name = request.json.get('name', None)
	description = request.json.get('description', None)
	sess.rename_feature_list(idx, name=name, description=description)
	return {}, 204

@app.put('/prompts/<int:idx>')
def rename_prompt(idx):
	name = request.json.get('name', None)
	description = request.json.get('description', None)
	sess.rename_prompt(idx, name=name, description=description)
	return {}, 204

@app.put('/prompts/<int:idx>/run')
def run_model_on_prompt_text(idx):
	text = request.json.get('text')
	tokens = sess.run_model_on_prompt_text(idx, text)
	return {'tokens': tokens}

@app.get('/prompts/<int:idx>/feature_list')
def get_cur_feature_list_idx_for_prompt(idx):
	idx = sess.get_cur_feature_list_idx_for_prompt(idx)
	return {'feature_list_id': idx}

@app.put('/prompts/<int:idx>/feature_list')
def change_cur_feature_list_for_prompt(idx):
	feature_list_idx = request.json['feature_list_idx']
	sess.change_cur_feature_list_for_prompt(idx, feature_list_idx)
	return {}, 204

@app.get('/feature_lists/<int:feature_list_idx>/features/<int:feature_idx>')
def get_feature_info_dict_by_idx(feature_list_idx, feature_idx):
	return get_feature_info_dict_by_idx(self, feature_list_idx, feature_idx)

@app.get('/feature_lists/<int:feature_list_idx>')
def get_feature_list_info(feature_list_idx):
	return json.dumps(sess.get_feature_list_info(feature_list_idx))

@app.post('/feature_lists/<int:feature_list_idx>/features')
def add_feature(feature_list_idx):
	feature_type = request.json['feature_type']
	name = request.json.get('name', None)
	description = request.json.get('description', None)
	if feature_type == 'observable':
		observable_tokens = request.json['observable_tokens']
		observable_weights = request.json['observable_weights']
		do_unembed_pullback = request.json.get('do_unembed_pullback', False)
		try:
			new_id = sess.add_feature_from_observable( feature_list_idx, observable_tokens, observable_weights, name=name, description=description, do_unembed_pullback=do_unembed_pullback)
		except Exception as e:
			return {'error': str(e)}, 400
	elif feature_type == 'sae':
		sae_idx = request.json['sae_idx']
		sae_feature_idx = request.json['sae_feature_idx']
		new_id = sess.add_feature_from_sae(feature_list_idx, sae_idx, sae_feature_idx, name=name, description=description)
	else:
		return {'error': f'Invalid feature type: {feature_type}'}, 400

	return {'feature_id': new_id}

@app.delete('/feature_lists/<int:feature_list_idx>/features/<int:feature_idx>')
def remove_feature_from_list(feature_list_idx, feature_idx):
	sess.remove_feature_from_list(feature_list_idx, feature_idx)
	return {}, 204

@app.put('/feature_lists/<int:feature_list_idx>/features/<int:feature_idx>')
def rename_feature(feature_list_idx, feature_idx):
	name = request.json.get('name', None)
	description = request.json.get('description', None)
	sess.rename_feature(feature_list_idx, feature_idx, name=name, description=description)
	return {}, 204

@app.get('/prompts/<int:prompt_idx>/feature_activs/feature_lists/<int:feature_list_idx>/features/<int:feature_idx>')
def get_feature_activs_on_prompt(prompt_idx, feature_list_idx, feature_idx):
	feature_activs = sess.get_feature_activs_on_prompt(prompt_idx, feature_list_idx, feature_idx)
	return feature_activs

@app.get('/prompts/<int:prompt_idx>/tokens/<int:token_pos>/feature_lists/<int:feature_list_idx>')
def get_feature_list_activs_on_token(prompt_idx, token_pos, feature_list_idx):
	return sess.get_feature_list_activs_on_token(prompt_idx, token_pos, feature_list_idx)

@app.get('/prompts/<int:prompt_idx>/tokens/<int:token_pos>/saes/<int:sae_idx>')
def get_sae_activs_on_token(prompt_idx, token_pos, sae_idx):
	k = request.args.get('k', None)
	if k is not None: k = int(k)
	return sess.get_sae_activs_on_token(prompt_idx, token_pos, sae_idx, k=k)

@app.put('/prompts/<int:prompt_idx>/tokens/<int:token_pos>/features/<int:feature_idx>/set_current_path')
def set_cur_comp_path_to_feature(prompt_idx, token_pos, feature_idx):
	sess.set_cur_comp_path_to_feature(prompt_idx, token_pos, feature_idx)
	return {}, 204

@app.get('/prompts/<int:prompt_idx>/comp_paths')
def list_comp_paths_for_prompt(prompt_idx):
	return sess.list_comp_paths_for_prompt(prompt_idx)

@app.delete('/prompts/<int:prompt_idx>/comp_paths/<path_idx>')
def delete_comp_path_by_id(prompt_idx, path_idx):
	sess.delete_comp_path_by_id(prompt_idx, path_idx)
	return {}, 204

@app.put('/prompts/<int:prompt_idx>/comp_paths/<path_idx>/nodes/<int:feature_pos>')
@app.put('/prompts/<int:prompt_idx>/comp_paths/<path_idx>', defaults={'feature_pos': -1})
def select_and_view_comp_path(prompt_idx, path_idx, feature_pos):
	top_k_children = request.args.get('k', 7)
	if path_idx == 'default': path_idx = None
	else: path_idx = int(path_idx)
	return sess.select_and_view_comp_path(prompt_idx, path_idx=path_idx, feature_pos=feature_pos, top_k_children=top_k_children)

@app.post('/prompts/<int:prompt_idx>/comp_paths/<path_idx>/nodes/<int:node_idx>/extend')
@app.post('/prompts/<int:prompt_idx>/comp_paths/<path_idx>/extend', defaults={'node_idx': -1})
def get_child_attrib_for_comp_path(prompt_idx, path_idx, node_idx):
	top_k_children = request.args.get('k', 7)
	if path_idx == 'default': path_idx = None
	else: path_idx = int(path_idx)

	return sess.get_child_attrib_for_comp_path(prompt_idx, request.json, path_idx=path_idx, top_k_children=top_k_children, extend=True, cur_node_idx=node_idx)

@app.post('/prompts/<int:prompt_idx>/comp_paths/<path_idx>/replace')
def replace_comp_path_with_component(prompt_idx, path_idx):
	top_k_children = request.args.get('k', 7)
	if path_idx == 'default': path_idx = None
	else: path_idx = int(path_idx)

	return sess.get_child_attrib_for_comp_path(prompt_idx, request.json, path_idx=path_idx, top_k_children=top_k_children, extend=False)


@app.post('/prompts/<int:prompt_idx>/comp_paths')
def save_cur_comp_path(prompt_idx):
	name = request.json.get('name', None)
	description = request.json.get('description', None)
	new_id = sess.save_cur_comp_path(prompt_idx, name=name, description=description)
	return {'comp_path_id': new_id}

@app.put('/prompts/<int:prompt_idx>/comp_paths/<int:path_idx>/rename')
def update_comp_path(prompt_idx, path_idx):
	name = request.json.get('name', None)
	description = request.json.get('description', None)
	sess.update_comp_path(prompt_idx, path_idx, name=name, description=description)
	return {}, 204

@app.delete('/prompts/<int:prompt_idx>/comp_paths/<int:path_idx>')
def remove_comp_path(prompt_idx, path_idx):
	sess.remove_comp_path(prompt_idx, path_idx)
	return {}, 204

@app.get('/prompts/<int:prompt_idx>/comp_paths/<path_idx>/nodes/<int:feature_pos>')
@app.get('/prompts/<int:prompt_idx>/comp_paths/<path_idx>', defaults={'feature_pos': -1})
def get_feature_info_from_comp_path(prompt_idx, path_idx, feature_pos):
	top_k_children = request.args.get('k', None)
	if path_idx == 'default': path_idx = None
	else: path_idx = int(path_idx)
	return sess.get_feature_info_from_comp_path(prompt_idx, path_idx=path_idx, feature_pos=feature_pos)

"""@app.post('/prompts/<int:prompt_idx>/comp_paths/<path_idx>/nodes/<int:feature_pos>/add_to_feature_list/<int:feature_list_idx>')
@app.post('/prompts/<int:prompt_idx>/comp_paths/<path_idx>/add_to_feature_list/<int:feature_list_idx>', defaults={'feature_pos': -1})
def add_feature_from_comp_path_to_feature_list(prompt_idx, path_idx, feature_list_idx, feature_pos):"""
@app.post('/prompts/<int:prompt_idx>/comp_paths/default/nodes/<int:feature_pos>/add_to_feature_list')
@app.post('/prompts/<int:prompt_idx>/add_feature_from_cur_comp_path', defaults={'feature_pos': -1})
def add_feature_from_comp_path_to_feature_list(prompt_idx, feature_pos):
	name = request.json.get('name', None)
	description = request.json.get('description', None)

	new_id = sess.add_feature_from_comp_path_to_feature_list(prompt_idx, None, None, feature_pos, name=name, description=description)
	return {'feature_id': new_id}

@app.put('/prompts/<int:prompt_idx>/comp_paths/default/nodes/<int:node_idx>/rename')
def update_comp_path_node(prompt_idx, node_idx):
	name = request.json.get('name', None)
	description = request.json.get('description', None)
	sess.update_comp_path_node(prompt_idx, node_idx, name, description)
	return {}, 204

"""@app.get('/prompts/<int:prompt_idx>/comp_paths/<path_idx>/nodes/<int:feature_pos>/input_invar_features')
@app.get('/prompts/<int:prompt_idx>/comp_paths/<path_idx>/input_invar_features', defaults={'feature_pos': -1})
def top_input_invar_features_from_comp_path(prompt_idx, path_idx, feature_pos):
	sae_idx = int(request.args['sae_idx'])
	k = int(request.args['k'])
	return sess.top_input_invar_features_from_comp_path(prompt_idx, path_idx, sae_idx, feature_pos=feature_pos, k=k)"""

@app.get('/feature_lists/<int:feature_list_idx>/features/<int:feature_idx>/input_invar_features')
def top_input_invar_features_from_feature_list(feature_list_idx, feature_idx):
	sae_idx = int(request.args['sae_idx'])
	k = int(request.args['k'])
	return sess.top_input_invar_features_from_feature_list(feature_list_idx, feature_idx, sae_idx, feature_pos=feature_pos, k=k)

@app.get('/prompts/<int:prompt_idx>/comp_paths/<path_idx>/nodes/<int:feature_pos>/deembeddings')
@app.get('/prompts/<int:prompt_idx>/comp_paths/<path_idx>/deembeddings', defaults={'feature_pos': -1})
def top_deembeddings_from_comp_path(prompt_idx, path_idx, feature_pos):
	k = int(request.args.get('k', 7))
	if path_idx == 'default': path_idx = None
	else: path_idx = int(path_idx)
	return sess.top_deembeddings_from_comp_path(prompt_idx, path_idx, feature_pos=feature_pos, k=k)

@app.get('/prompts/<int:prompt_idx>/comp_paths/<path_idx>/nodes/<int:feature_pos>/input_invar')
@app.get('/prompts/<int:prompt_idx>/comp_paths/<path_idx>/input_invar', defaults={'feature_pos': -1})
def top_input_invar_features_from_comp_path(prompt_idx, path_idx, feature_pos):
	k = int(request.args.get('k', 7))
	sae_idx = int(request.args.get('sae_idx', 0))

	if path_idx == 'default': path_idx = None
	else: path_idx = int(path_idx)
	
	return sess.top_input_invar_features_from_comp_path(prompt_idx, path_idx, sae_idx, feature_pos=feature_pos, k=k)

@app.get('/feature_lists/<int:feature_list_idx>/features/<int:feature_idx>/deembeddings')
def top_deembeddings_from_feature_list(feature_list_idx, feature_idx):
	sae_idx = int(request.args['sae_idx'])
	k = int(request.args['k'])
	return sess.top_deembeddings_from_feature_list(feature_list_idx, feature_idx, sae_idx, feature_pos=feature_pos, k=k)

import traceback
@app.post('/session/load')
def load_session():
	path = request.json['path']
	try:
		sess.load_wrapper(path)
	except Exception as e:
		return {"error": traceback.format_exc()}, 500
	return {}, 204

@app.post('/session/save')
def save_session():
	path = request.json['path']
	sess.save_wrapper(path, save_feature_tensors=request.json.get('save_feature_tensors', True))
	return {}, 204

@app.put('/session/rename')
def rename_session():
	name = request.json.get('name', None)
	description = request.json.get('description', None)
	if name is not None: sess.name = name
	if description is not None: sess.description = description
	return {}, 204

@app.get("/session")
def get_session_details():
	return {'name': sess.name, 'description': sess.description}

if __name__ == '__main__':
	# the port number corresponds to the word "intrp" when input on a cell phone keypad
	app.jinja_env.auto_reload = True
	app.config['TEMPLATES_AUTO_RELOAD'] = True

	app.run(host="0.0.0.0", port=46877)
