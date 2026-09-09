PYTHON ?= python3
SAGE_PYTHON ?= sage -python
CC ?= cc
CXX ?= c++

.PHONY: all locators recover verify reproduce test tools cuda rust clean

all: locators verify

locators:
	$(PYTHON) scripts/recover_locators.py

recover: locators
	$(SAGE_PYTHON) scripts/recover_key.py

verify:
	$(SAGE_PYTHON) verify_key.sage

reproduce: recover
	$(SAGE_PYTHON) verify_key.sage --key build/recovered_key.json

test: tools
	$(PYTHON) -m unittest discover -s tests -v

tools: build/sequence-to-cado build/extract-relations

build/sequence-to-cado: scripts/sequence_to_cado.c
	mkdir -p build
	$(CC) -O3 -std=c11 -Wall -Wextra -Wpedantic $< -o $@

build/extract-relations: scripts/extract_relations.cpp
	mkdir -p build
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Wpedantic $< -o $@

cuda:
	$(MAKE) -C cuda

rust:
	cargo test --manifest-path rust/Cargo.toml --release

clean:
	rm -rf build cuda/bin rust/target
	find python scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +
