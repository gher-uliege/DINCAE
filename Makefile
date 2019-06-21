
docs/index.html: DINCAE.py
	pdoc --overwrite --html DINCAE.py; mv DINCAE.m.html docs/index.html
