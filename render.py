import sys
import os
import jinja2

if __package__ is not None and __package__ != "":
	from . import tc_peek_backend as backend 
else:
	import tc_peek_backend as backend

def render_sess(sess, outpath="output.html", template_path="templates/render_template.html"):
	with open(template_path) as ifp:
		s = ifp.read()

	env = jinja2.Environment(loader=jinja2.FileSystemLoader('.'), autoescape=True)
	env.globals.update(dict(
		sess=sess,
		# here be python functions
		min=min,
		max=max
	))

	template = env.from_string(s)
	rendered = template.render()

	with open(outpath, "w") as ofp:
		ofp.write(rendered)

if __name__ == "__main__":
	inpath = sys.argv[1]
	sess = backend.Session()
	sess.load_wrapper(inpath, load_tensors=False)
	
	if len(sys.argv) > 2:
		outpath = sys.argv[2]
	
	render_sess(sess, outpath=outpath)
