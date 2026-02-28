# Makefile for LunarLander DQN Project
# Automate common tasks such as installation, training, evaluation, testing, cleaning, Docker operations, linting, and formatting.

.PHONY: help install train evaluate test unit-test clean clean-artifacts docker-build docker-train docker-eval lint format

# Default virtual environment (optional override: make VENV=myenv)
VENV ?= venv
PYTHON := python

help:
	@echo "LunarLander DQN - Available Commands"
	@echo "===================================="
	@echo "setup			- Run setup.sh to initialize the environment."
	@echo "install        	- Install dependencies"
	@echo "train          	- Train DQN agent"
	@echo "evaluate       	- Evaluate trained agent"
	@echo "test           	- Run single rendered test episode"
	@echo "unit-test	  	- Run pytest unit tests"
	@echo "convert-gifs		- Convert MP4 videos to GIFs"
	@echo "clean          	- Remove Python cache files"
	@echo "clean-artifacts	- Remove generated models/plots/videos/logs"
	@echo "docker-build   	- Build Docker image"
	@echo "docker-train   	- Train using Docker"
	@echo "docker-eval    	- Evaluate using Docker"
	@echo "lint           	- Run flake8 on entire project"
	@echo "format         	- Format project using black"

setup:
	setup:
	chmod +x setup.sh && ./setup.sh
	
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	mkdir -p models plots videos logs assets/videos

train:
	$(PYTHON) train.py

evaluate:
	$(PYTHON) evaluate.py --model ./models/best_model.pth --episodes 100

test:
	$(PYTHON) evaluate.py --model ./models/best_model.pth --test --render

unit-test:
	pytest -v --tb=short

convert-gifs:
	$(PYTHON) convert_videos.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-artifacts:
	rm -rf models/*.pth
	rm -rf plots/*.png
	rm -rf videos/*
	rm -rf assets/videos/*.gif
	rm -rf logs/*.log

docker-build:
	docker-compose build

docker-train:
	docker-compose up --abort-on-container-exit dqn-training

docker-eval:
	docker-compose --profile evaluation up --abort-on-container-exit dqn-evaluation

lint:
	flake8 . --max-line-length=100

format:
	black . --line-length=100