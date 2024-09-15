SHELL := /bin/bash

.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

format: ## Format the code with the Ruff tool.
	ruff format

lint: ## Lint the code with the Ruff tool.
	ruff check

isort: ## Sort the imports with the Ruff tool.
	ruff check --select I --fix .

run-chain: ## Run the LLM chain example.
	python chain.py

run-chat: ## Run the LLM chat example.
	python chat.py

run-storing-history: ## Run the LLM storing history example.
	python storing_history.py

run-agent: ## Run the LLM agent example.
	python agent.py

run-retriever: ## Run the LLM retriever example.
	python retriever.py

run-movie-expert: ## Run the LLM movie expert example.
	python movie_expert.py