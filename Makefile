.PHONY: doctor build build-host-io check package validate-release-local

doctor:
	python scripts/xlog_doctor.py

build:
	cargo build --release

build-host-io:
	cargo build --release -p xlog-cli --features host-io

check:
	cargo test --workspace --all-targets --exclude pyxlog

package:
	cargo build --release -p xlog-cli

validate-release-local:
	python scripts/xlog_doctor.py --workflow prob-cli
