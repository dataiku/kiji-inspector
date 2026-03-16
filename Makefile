.PHONY: format lint fix bump-patch bump-minor bump-major bump-dry-run

format: ## Format code with ruff
	uvx ruff format .

lint: ## Run linter
	uvx ruff check src/ utils/

fix: ## Auto-fix lint issues and format
	uvx ruff check src/ utils/ --fix
	uvx ruff format .

bump-patch: ## Bump patch version (0.0.x)
	uvx --from commitizen cz bump --increment PATCH

bump-minor: ## Bump minor version (0.x.0)
	uvx --from commitizen cz bump --increment MINOR

bump-major: ## Bump major version (x.0.0)
	uvx --from commitizen cz bump --increment MAJOR

bump-dry-run: ## Preview what the next bump would do
	uvx --from commitizen cz bump --dry-run
