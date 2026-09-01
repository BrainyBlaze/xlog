from __future__ import annotations

import os

import pytest


torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")


def _require_cuda() -> None:
    if torch.cuda.is_available():
        return
    if os.environ.get("XLOG_REQUIRE_CUDA") == "1":
        raise RuntimeError("XLOG_REQUIRE_CUDA=1 but PyTorch cannot access CUDA")
    pytest.skip("CUDA is unavailable")


def test_compile_file_resolves_transitive_modules_with_native_resolver(tmp_path) -> None:
    _require_cuda()

    (tmp_path / "facts.xlog").write_text("source(7).\n", encoding="utf-8")
    (tmp_path / "rules.xlog").write_text(
        "use facts.\nresult(X) :- source(X).\n",
        encoding="utf-8",
    )
    entrypoint = tmp_path / "main.xlog"
    entrypoint.write_text("use rules.\n?- result(X).\n", encoding="utf-8")

    program = pyxlog.LogicProgram.compile_file(
        entrypoint,
        module_paths=[tmp_path],
        device=0,
        memory_mb=512,
    )
    evaluated = program.evaluate()

    assert len(evaluated.queries) == 1
    query = evaluated.queries[0]
    assert query.relation_name == "__xlog_query_0"
    values = torch.utils.dlpack.from_dlpack(query.tensors[0]).cpu().tolist()
    assert values == [7]
