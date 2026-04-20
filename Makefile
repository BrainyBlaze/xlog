ACTIONLINT ?= actionlint
SHELLCHECK ?= shellcheck
PACKAGE_OUTPUT ?= dist

.PHONY: doctor build build-host-io check package validate-release-local lint-workflows lint-shell check-tracked-ignored

doctor:
	python scripts/xlog_doctor.py

# Local workflow validation mirrors .github/workflows/ci.yml:
# `actionlint` consumes `shellcheck` from PATH to lint embedded bash in workflow `run:` blocks.
lint-workflows:
	@bash -lc 'set -euo pipefail; \
	if ! command -v "$(ACTIONLINT)" >/dev/null 2>&1; then \
		echo "actionlint is required for make lint-workflows." >&2; \
		echo "Install it with the same version CI uses, for example:" >&2; \
		echo "  ACTIONLINT_VERSION=1.7.7" >&2; \
		echo "  curl -sSL https://github.com/rhysd/actionlint/releases/download/v\$$ACTIONLINT_VERSION/actionlint_\$$ACTIONLINT_VERSION_linux_amd64.tar.gz | tar -xz actionlint" >&2; \
		echo "  install actionlint /usr/local/bin/actionlint" >&2; \
		exit 127; \
	fi; \
	if ! command -v "$(SHELLCHECK)" >/dev/null 2>&1; then \
		echo "shellcheck is required because actionlint validates embedded bash with it." >&2; \
		echo "Install it with your package manager, for example: sudo apt-get install -y shellcheck" >&2; \
		exit 127; \
	fi; \
	PATH="$$(dirname "$(SHELLCHECK)"):$$PATH" "$(ACTIONLINT)" -color'

lint-shell:
	@bash -lc 'set -euo pipefail; \
	if ! command -v "$(SHELLCHECK)" >/dev/null 2>&1; then \
		echo "shellcheck is required for make lint-shell." >&2; \
		echo "Install it with your package manager, for example: sudo apt-get install -y shellcheck" >&2; \
		exit 127; \
	fi; \
	files=$$(rg --files scripts .github -g '"'"'*.sh'"'"'); \
	if [ -z "$$files" ]; then \
		echo "No shell scripts found."; \
		exit 0; \
	fi; \
	printf "%s\n" "$$files"; \
	$(SHELLCHECK) $$files'

check-tracked-ignored:
	@bash -lc 'set -euo pipefail; \
	files=$$(git ls-files -ci --exclude-standard); \
	if [ -z "$$files" ]; then \
		echo "No tracked files match .gitignore rules."; \
		exit 0; \
	fi; \
	echo "Tracked files must not match .gitignore rules:"; \
	printf "%s\n" "$$files"; \
	exit 1'

build:
	cargo build --release

build-host-io:
	cargo build --release -p xlog-cli --features host-io

check:
	cargo test --workspace --all-targets --exclude pyxlog

package:
	bash scripts/package_cli_release.sh --output $(PACKAGE_OUTPUT)

validate-release-local:
	bash scripts/validate_release_gpu.sh --mode release
