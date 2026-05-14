# Makefile for quick-run commands

MKFILE_PATH:=$(abspath $(lastword $(MAKEFILE_LIST)))
ABS_PATH:=$(dir $(MKFILE_PATH))
CURRENT_DIR:=$(notdir $(patsubst %/,%,$(dir $(MKFILE_PATH))))

ifeq ($(OS),Windows_NT)
	ACTIVATE_CMD:=$(ABS_PATH).venv\Scripts\Activate.ps1
else
	ACTIVATE_CMD:=source $(ABS_PATH).venv/bin/activate
endif

.PHONY: init install

.ONESHELL:
init:
	python -m venv .venv
	$(ACTIVATE_CMD)
	pip install --upgrade pip
	pip install -r requirements.txt

install:
	pip install -r requirements.txt
