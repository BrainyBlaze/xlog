"""Type stubs for the top-level ``pyxlog`` package.

Everything is re-exported from :mod:`pyxlog._native`; this file exists so that
``import pyxlog`` surfaces the same names as ``from pyxlog._native import *``.
"""

from __future__ import annotations

# Re-export the full public surface of the native module.
from pyxlog._native import (
    # Module constant
    __version__ as __version__,
    # Logic (pure Datalog)
    LogicProgram as LogicProgram,
    CompiledLogicProgram as CompiledLogicProgram,
    LogicRelationSession as LogicRelationSession,
    LogicQueryResult as LogicQueryResult,
    LogicEvalResult as LogicEvalResult,
    # Probabilistic / neural-symbolic
    Program as Program,
    CompiledProgram as CompiledProgram,
    EvalResult as EvalResult,
    McDeviceEvalResult as McDeviceEvalResult,
    # Training
    EpochStats as EpochStats,
    TrainingHistory as TrainingHistory,
    train_model as train_model,
    train_model_tensor as train_model_tensor,
    # ILP
    IlpProgramFactory as IlpProgramFactory,
    CompiledIlpProgram as CompiledIlpProgram,
    IlpTaggedCreditDeviceResult as IlpTaggedCreditDeviceResult,
    # DLPack / Arrow utilities
    dlpack_roundtrip as dlpack_roundtrip,
)

# Arrow imports are feature-gated; expose them for type checkers but they may
# be absent at runtime when pyxlog is built without ``arrow-device-import``.
try:
    from pyxlog._native import (
        export_arrow_device as export_arrow_device,
        import_arrow_device as import_arrow_device,
    )
except ImportError:
    pass
