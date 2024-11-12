# [Peek](https://peekai.org): a tool and platform for peeking inside large language models

Peek is a program that lets you investigate why LLMs do what they do without writing any code. If you are interested in a specific behavior of an LLM -- e.g. "How does my model know that the capital of England is London and not Paris", or "What would cause my model to use the word 'thou' and not 'you'?" -- then with a couple of clicks, you can find *features* inside the model that are responsible for that behavior. 

Peek also comes with a companion website: [peekai.org](https://peekai.org). Once you've performed an investigation, you can share your findings on [peekai.org](https://peekai.org) and view other people's findings there as well. 

# Installation

To install Peek, first make sure that Python 3 >= 3.10 is installed on your machine. Then, download this repository:

``git clone https://github.com/jacobdunefsky/peek.git peek``

Enter the downloaded repository with `cd peek`.

To avoid dependency conflicts with your other Python packages, you might want to install Peek into a virtual environment. To do so, type the following commands:

``python3 -m venv peek_venv``

``source peek_venv/bin/activate``

This will activate the new virtual environment.

Now, install the required Python packages:

``python3 -m pip install -r requirements.txt``

You now should be good to go!

# Running Peek

To run the Peek server, `cd` into the directory where you installed Peek.

If you installed Peek with a virtual environment, then run the command

``source peek_venv/bin/activate``

Now, run the following command:

``python3 server.py``

This will launch the Peek server, which will listen on port 46877.

Then, in your favorite web browser, navigate to `localhost:46877` to access Peek.

# Documentation

A detailed walkthrough of how to use Peek's main features is provided in [walkthrough.md](docs/walkthrough.md).

A tutorial on how to perform feature steering with Peek is provided in [steering\_tutorial.md](docs/steering_tutorial.md).

# Contact

This program, and the associated website, is created/maintained by [Jacob Dunefsky](jacobdunefsky.github.io). To get in contact, drop me an email at `jhunter [dot] dunefsky [at] gmail [dot] com`.

# Technical note

Techically speaking, Peek is a tool for investigating *transcoder feature circuits* corresponding to specific *observables*. The underlying methods most directly come from my (Jacob Dunefsky's) previous research on [observables](https://openreview.net/pdf?id=ETNx4SekbY) and [transcoder feature circuits](https://arxiv.org/pdf/2406.11944); these methods, in turn, are built upon the ever-growing body of high-quality research coming from the mechanistic interpretability community. In particular, these methods come from a lineage of research on sparse autoencoders (SAEs) and circuit discovery. 
