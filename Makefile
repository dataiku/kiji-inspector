.PHONY: format lint fix

format: ## Format code with ruff
	uvx ruff format .

lint: ## Run linter
	uvx ruff check src/

fix: ## Auto-fix lint issues and format
	uvx ruff check src/ --fix
	uvx ruff format .
