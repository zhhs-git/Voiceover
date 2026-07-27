.PHONY: help install worker-install setup dev desktop worker web web-dev restart stop clean test test-ts test-py lint format build

# Default target
.DEFAULT_GOAL := help

# Paths
DESKTOP_DIR := apps/desktop
WORKER_DIR  := workers/python

# Colors
GREEN  := \033[0;32m
YELLOW := \033[1;33m
RED    := \033[0;31m
NC     := \033[0m # No Color

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-18s$(NC) %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────────

install: ## Install all dependencies (Node + Python)
	@echo "$(YELLOW)Installing Node dependencies...$(NC)"
	cd $(DESKTOP_DIR) && npm install
	@echo ""
	@echo "$(YELLOW)Installing Python worker dependencies...$(NC)"
	cd $(WORKER_DIR) && uv sync
	@echo ""
	@echo "$(GREEN)✓ All dependencies installed$(NC)"

node-install: ## Install only Node dependencies
	cd $(DESKTOP_DIR) && npm install

worker-install: ## Install only Python worker dependencies
	cd $(WORKER_DIR) && uv sync

setup: install ## Full setup (alias for install)

# ── Development ────────────────────────────────────────────────────────

dev: ## Start all services (desktop + worker)
	@echo "$(GREEN)Starting desktop app (Vite dev server)...$(NC)"
	@echo "$(YELLOW)Python worker is spawned as a subprocess by Tauri.$(NC)"
	@echo "$(YELLOW)Run 'make desktop' for Vite-only, or 'make tauri' for full Tauri.$(NC)"
	cd $(DESKTOP_DIR) && npm run tauri dev

desktop: ## Start Vite dev server only (frontend without Tauri shell)
	cd $(DESKTOP_DIR) && npm run dev

tauri: ## Start full Tauri desktop app (frontend + Rust shell + Python worker)
	cd $(DESKTOP_DIR) && npm run tauri dev

worker: ## Run Python worker standalone (for debugging)
	cd $(WORKER_DIR) && uv run python -m audiobook_worker

web: ## Build and start the shared LAN web application
	npm run web

web-dev: ## Start the LAN web API for frontend development
	npm run web:dev

# ── Restart ────────────────────────────────────────────────────────────

restart: stop dev ## Stop then start all services

restart-desktop: stop desktop ## Stop then start Vite dev server only

restart-tauri: stop tauri ## Stop then start full Tauri app

# ── Stop ───────────────────────────────────────────────────────────────

stop: ## Kill all running project processes
	@echo "$(YELLOW)Stopping audiobook-generator processes...$(NC)"
	@-pkill -f "vite" 2>/dev/null || true
	@-pkill -f "tauri dev" 2>/dev/null || true
	@-pkill -f "tauri-cli" 2>/dev/null || true
	@-pkill -f "audiobook_worker" 2>/dev/null || true
	@echo "$(GREEN)✓ Stopped$(NC)"

# ── Clean ──────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf $(DESKTOP_DIR)/dist
	rm -rf $(DESKTOP_DIR)/src-tauri/target
	rm -rf $(WORKER_DIR)/.pytest_cache
	rm -rf $(WORKER_DIR)/audiobook_worker/__pycache__
	find $(WORKER_DIR) -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(NC)"

distclean: clean ## Deep clean including node_modules and venvs
	@echo "$(YELLOW)Deep cleaning...$(NC)"
	rm -rf node_modules
	rm -rf $(DESKTOP_DIR)/node_modules
	rm -rf $(WORKER_DIR)/.venv
	rm -rf packages/shared/node_modules
	@echo "$(GREEN)✓ Deep cleaned$(NC)"

# ── Testing ────────────────────────────────────────────────────────────

test: ## Run all tests (TypeScript + Python)
	@echo "$(YELLOW)Running TypeScript tests...$(NC)"
	cd $(DESKTOP_DIR) && npm test -- --run 2>&1 || true
	@echo ""
	@echo "$(YELLOW)Running Python tests...$(NC)"
	cd $(WORKER_DIR) && uv run pytest 2>&1 || true
	@echo ""
	@echo "$(GREEN)✓ All tests complete$(NC)"

test-ts: ## Run TypeScript tests only
	cd $(DESKTOP_DIR) && npm test -- --run

test-ts-watch: ## Run TypeScript tests in watch mode
	cd $(DESKTOP_DIR) && npm test

test-py: ## Run Python tests only
	cd $(WORKER_DIR) && uv run pytest

test-py-watch: ## Run Python tests in watch mode
	cd $(WORKER_DIR) && uv run ptw

# ── Code Quality ───────────────────────────────────────────────────────

lint: ## Run all linters
	cd $(DESKTOP_DIR) && npx tsc --noEmit
	cd $(WORKER_DIR) && uv run --extra dev ruff check .

format: ## Format all code
	cd $(DESKTOP_DIR) && npx prettier --write 'src/**/*.{ts,tsx,css}'
	cd $(WORKER_DIR) && uv run --extra dev ruff format .

# ── Build ───────────────────────────────────────────────────────────────

build: ## Build desktop app for production
	cd $(DESKTOP_DIR) && npm run build
	cd $(DESKTOP_DIR) && npm run tauri build
