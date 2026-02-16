.PHONY: format lint fix

format: ## Format code with ruff
	uv run ruff format .

lint: ## Run linter
	uv run ruff check src/

fix: ## Auto-fix lint issues and format
	uv run ruff check src/ --fix
	uv run ruff format .
