# Rust API

Rust reference pages are generated with:

```bash
cargo doc --workspace --no-deps --locked
```

The generated rustdoc output is published under:

- [Rust workspace reference](generated/rust/index.html)

If this link shows placeholder content in a local fast build, run `make docs` in an environment with the Rust and CUDA toolchains available.

