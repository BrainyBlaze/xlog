FROM ghcr.io/astral-sh/uv@sha256:9a853e40f2eb00f980d265a84fba919dc38ff17750197e7f13c055e4e2f12edd AS uv
FROM nvidia/cuda@sha256:69e9e39eb8fe2cda271654a0f5eac2f1bb946b2fb9c460eb19c7c3c155f4e64e

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl patchelf pkg-config git build-essential libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ADD --checksum=sha256:231fcae47ee950621374725b5fc3b5441d2fbda2955ac45dd7dc816d8876cfe7 https://static.rust-lang.org/dist/2026-10-15/rust-1.99.1-x86_64-unknown-linux-gnu.tar.xz /tmp/rust.tar.xz
RUN mkdir /tmp/rust-distribution \
    && tar --strip-components=1 -xJf /tmp/rust.tar.xz -C /tmp/rust-distribution \
    && /tmp/rust-distribution/install.sh --prefix=/opt/rust --without=rust-docs \
    && rm -rf /tmp/rust-distribution /tmp/rust.tar.xz

COPY --from=uv /uv /usr/local/bin/uv
RUN uv python install --install-dir /opt/xlog-python 3.10.22 \
    && uv venv --python /opt/xlog-python/cpython-3.10.22-linux-x86_64-gnu/bin/python3.10 /opt/xlog-build-env \
    && uv pip install --python /opt/xlog-build-env/bin/python --only-binary :all: maturin==1.15.0

ENV PATH="/opt/rust/bin:/opt/xlog-build-env/bin:${PATH}" \
    CARGO_HOME=/cache/cargo-home \
    CARGO_TARGET_DIR=/cache/target \
    CARGO_BUILD_JOBS=2 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /src
ENTRYPOINT ["timeout", "--signal=TERM", "--kill-after=30s", "7200s", "python", "/src/scripts/install_pyxlog_for_python.py", "--build-only", "--python", "/opt/xlog-build-env/bin/python", "--compatibility", "manylinux_2_34", "--wheel-dir", "/out"]
