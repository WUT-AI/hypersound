.DEFAULT_GOAL := check

# Code linting
.PHONY: lint
lint:
	@echo "\n>>> Default linting checks"
	@flake8
	@isort --check --diff --color .
	@black --check --diff --color .

# mypy checks
.PHONY: mypy
mypy:
	@echo "\n>>> mypy checks"
	@mypy *.py hypersound

# Extended linting
.PHONY: qa
qa:
	@echo "\n>>> Extended linting checks"
	@pylint --disable=all --enable=duplicate-code hypersound --exit-zero
	@flake8 --ignore= --select=D --exit-zero
	@flake8 --ignore= --select=CC --exit-zero
	@flake8 --ignore= --select=TAE --exit-zero

# Default code check
.PHONY: check
check: lint mypy qa;



