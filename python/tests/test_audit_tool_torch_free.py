"""The section F audit tool must run on a torch-free interpreter: its own
work is XML parsing plus frame arithmetic, and torch is incidental to it
(it used to arrive transitively through `caviar_continuous`'s module-level
``import torch``). Same requirement for the census tool's import chain.

The check runs in a subprocess with torch blocked at the import machinery,
so it is meaningful on every environment -- torch installed or not --
which is also why this file deliberately has NO ``importorskip("torch")``:
it belongs to the torch-free subset of the caviar suites.
"""
import subprocess
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "caviar_woled"

_SCRIPT = r'''
import sys


class _TorchBlocker:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is blocked for this audit-tool test")


sys.meta_path.insert(0, _TorchBlocker())
sys.path.insert(0, sys.argv[1])

# The audit tool itself plus every import its pipeline performs lazily
# (`main` and the event builders import exactly these names).
import audit_dump_vs_xml
from caviar_continuous import _person_num, load_continuous
from caviar_xml_corpus import _canonical_pair, load_xml_corpus

# The census tool's own chain, same torch-free requirement.
import xml_meeting_census
from run_caviar_cv import _xml_family_fold_assignment, _xml_stem

# A representative torch-free computation, not just an import.
assert audit_dump_vs_xml.pair_transition_events([0, 40, 80, 120], {40, 80}) == [
    (40, "init"), (120, "term"),
]
print("torch-free audit chain OK")
'''


def test_audit_and_census_import_chains_run_without_torch():
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(EXAMPLE_DIR)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert "torch-free audit chain OK" in proc.stdout
