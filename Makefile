
docs/index.html: DINCAE/__init__.py
	pdoc --overwrite --html DINCAE; mv DINCAE/index.html docs/index.html
