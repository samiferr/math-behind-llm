PY ?= python3

.PHONY: all cover epub pdf publish new-edition new-update new-revision clean

all:            ## Build cover, EPUB and PDF into dist/
	$(PY) scripts/build.py cover epub pdf

cover:
	$(PY) scripts/build.py cover

epub:
	$(PY) scripts/build.py epub

pdf:
	$(PY) scripts/build.py pdf

publish:        ## Build and copy the files into editions/<Nth>-edition/
	$(PY) scripts/build.py cover epub pdf --publish

new-edition:    ## 1.x.x -> 2.0.0   usage: make new-edition NOTE="..."
	$(PY) scripts/new_edition.py edition --note "$(NOTE)"

new-update:     ## 1.0.x -> 1.1.0
	$(PY) scripts/new_edition.py update --note "$(NOTE)"

new-revision:   ## 1.0.0 -> 1.0.1
	$(PY) scripts/new_edition.py revision --note "$(NOTE)"

clean:
	rm -rf build dist
