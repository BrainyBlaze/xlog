ACTIONLINT ?= actionlint
SHELLCHECK ?= shellcheck
PACKAGE_OUTPUT ?= dist

.PHONY: doctor build build-host-io check package validate-release-local lint-workflows lint-shell

doctor:
	python scripts/xlog_doctor.py

# Local workflow validation mirrors .github/workflows/ci.yml:
# `actionlint` consumes `shellcheck` from PATH to lint embedded bash in workflow `run:` blocks.
lint-workflows:
	PATH="$$(dirname "$(SHELLCHECK)"):$$PATH" $(ACTIONLINT) -color

lint-shell:
	@bash -lc 'set -euo pipefail; \
	files=$$(rg --files scripts .github -g '"'"'*.sh'"'"'); \
	if [ -z "$$files" ]; then \
		echo "No shell scripts found."; \
		exit 0; \
	fi; \
	printf "%s\n" "$$files"; \
	$(SHELLCHECK) $$files'

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
