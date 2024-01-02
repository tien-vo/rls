NAME := rls
CONDA_LOCK := conda-lock.yml
POETRY_LOCK := poetry.lock

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "TODO: Edit help string"

.PHONY: install
install: pyproject.toml $(POETRY_LOCK) $(CONDA_LOCK)
	@if [ -z $(shell command -v micromamba 2> /dev/null) ]; then \
	echo "Micromamba binary not found"; \
	exit 1; \
	fi
	@echo "Creating virtual environment from $(CONDA_LOCK) ..."
	@micromamba create --quiet --yes --override-channels --name $(NAME) --file $(CONDA_LOCK)
	@echo "Installing packages from $(POETRY_LOCK) ..."
	@micromamba run -n $(NAME) poetry install
	@echo "Done installation!"

.PHONY: format
format:
	@micromamba run -n $(NAME) poetry run isort src/
	@micromamba run -n $(NAME) poetry run black src/

.PHONY: test
test:
	@micromamba run -n $(NAME) pytest -s

.PHONY: clean
clean:
	find . | grep -E "(/__pycache__$$)" | xargs rm -rf
