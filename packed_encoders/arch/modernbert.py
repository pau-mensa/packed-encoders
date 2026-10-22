"""ModernBERT adapter and packed engine over the existing execution schedules."""

from __future__ import annotations

from typing import Any, Mapping

from packed_encoders.runtime.cuda_epilogue import CUDAEpilogueSupport
from packed_encoders.engine import Capabilities, ModelBinding, ValidationResult
from packed_encoders.errors import PackedEncodersError
from packed_encoders.locate import is_modernbert_encoder
from packed_encoders.arch.modernbert_pieces import ModernBertPieces


def _check_options(options):
    from packed_encoders.graph import GraphConfig
    from packed_encoders.train_graph import TrainGraphConfig

    unknown = options.keys() - {"cuda_graph", "train_cuda_graph", "cuda_graph_seq_cutoff", "attention_backend", "validate"}
    if unknown:
        raise PackedEncodersError(f"unsupported ModernBERT options: {sorted(unknown)}")
    for name, config_type in (("cuda_graph", GraphConfig), ("train_cuda_graph", TrainGraphConfig)):
        value = options.get(name)
        if value is not None and not isinstance(value, (bool, config_type)):
            raise PackedEncodersError(f"{name} requires a bool or {config_type.__name__}")
    cutoff = options.get("cuda_graph_seq_cutoff")
    if cutoff is not None and (type(cutoff) is not int or cutoff <= 0):
        raise PackedEncodersError("cuda_graph_seq_cutoff must be a positive integer or None")


class ModernBertAdapter:
    name = "hf-modernbert"

    def bind(self, module: object) -> ModelBinding | None:
        if is_modernbert_encoder(module):
            return ModelBinding(self, module, module)
        return None

    def install(self, binding: ModelBinding, packed: PackedModernBert) -> None:
        from packed_encoders.arch.modernbert_cpu import PackedModernBertCPU
        if isinstance(packed, PackedModernBertCPU):
            packed.install_forward()
            return
        from packed_encoders.pack import _make_forward
        from packed_encoders.state import ATTR

        setattr(binding.patch_target, ATTR, packed.state)
        binding.patch_target.forward = _make_forward(binding.patch_target, packed.state)


class ModernBert:
    name = "modernbert"
    adapters = (ModernBertAdapter(),)

    def __init__(self, *, pieces: ModernBertPieces | None = None, device="auto"):
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
        self.pieces = pieces
        self.device = device

    def _cpu(self, binding):
        return self.device == "cpu" or (self.device == "auto" and next(binding.weight_source.parameters()).device.type == "cpu")

    def prepare(self, binding: ModelBinding, options: Mapping[str, Any]):
        if self._cpu(binding):
            if self.pieces is not None:
                raise PackedEncodersError("CUDA operation pieces cannot be used by the CPU engine")
            from packed_encoders.arch.modernbert_cpu import ModernBertCPU
            return ModernBertCPU().prepare(binding, options)
        from packed_encoders.pack import _prepare_modernbert

        _check_options(options)
        state = _prepare_modernbert(binding.weight_source, pieces=self.pieces, **options)
        return PackedModernBert(binding, state)

    def validate(self, binding: ModelBinding, **kwargs: Any) -> ValidationResult:
        if self._cpu(binding):
            if self.pieces is not None:
                raise PackedEncodersError("CUDA operation pieces cannot be used by the CPU engine")
            from packed_encoders.arch.modernbert_cpu import ModernBertCPU
            return ModernBertCPU().validate(binding, **kwargs)
        from packed_encoders.validate import _validate_modernbert

        return ValidationResult(self.name, _validate_modernbert(binding.weight_source, pieces=self.pieces, **kwargs))


class PackedModernBert(CUDAEpilogueSupport):
    capabilities = Capabilities(training=True, inference_capture=True, training_capture=True)
    def __init__(self, binding: ModelBinding, state):
        self.binding = binding
        self.state = state
        self.composition = state.pieces
        self.pieces = self.composition.all()
        self._closed = False
        self._init_epilogues()

    def _require_open(self):
        if self._closed:
            raise PackedEncodersError("this packed engine has been closed; pack the model again")

    @property
    def graph_enabled(self) -> bool:
        return self.state.graph_enabled

    @graph_enabled.setter
    def graph_enabled(self, enabled: bool) -> None:
        self.state.graph_enabled = enabled

    def forward_packed(self, batch):
        """No metadata conversion or device reads; retain direct packed_forward ABI.

        Caller supplies nonempty contiguous sequences, consistent int32 boundaries,
        and HF-compatible positions in [0, max_seqlen). Value validation belongs at
        collation, outside timed execution. Tensor metadata checks do not synchronize.
        """
        import torch
        from packed_encoders.forward import packed_forward

        self._require_open()
        ids, cu, positions = batch.input_ids, batch.cu_seqlens, batch.position_ids
        if cu is None or positions is None or batch.max_seqlen is None:
            raise PackedEncodersError("ModernBERT requires cu_seqlens, max_seqlen and position_ids")
        if ids.ndim != 1 or positions.shape != ids.shape or cu.ndim != 1 or cu.numel() < 2:
            raise PackedEncodersError("expected flat IDs/positions and sequence boundaries of shape [B+1]")
        if cu.dtype != torch.int32 or ids.dtype != torch.int64 or positions.dtype != torch.int64:
            raise PackedEncodersError("ModernBERT requires int64 IDs/positions and int32 cu_seqlens")
        if ids.device != cu.device or ids.device != positions.device:
            raise PackedEncodersError("packed metadata must be on the token device")
        if ids.numel() == 0 or batch.max_seqlen <= 0:
            raise PackedEncodersError("empty ModernBERT packed batches are unsupported")
        return packed_forward(
            self.binding.weight_source, self.state.params, ids, cu,
            batch.max_seqlen, positions, backend=self.state.attention_backend,
        )

    def validate(self, **kwargs: Any) -> ValidationResult:
        self._require_open()
        return ModernBert(pieces=self.composition).validate(self.binding, **kwargs)

    def configure(self, options: Mapping[str, Any]) -> None:
        """Preserve repeat-pack graph enabling; backend switches require unpack."""
        from packed_encoders.pack import _enable_graphs, _enable_train_graphs

        self._require_open()
        _check_options(options)
        backend = options.get("attention_backend")
        if backend is not None and backend != self.state.attention_backend:
            raise PackedEncodersError("unpack before changing the prepared attention backend")
        graph, train = options.get("cuda_graph"), options.get("train_cuda_graph")
        cutoff = options.get("cuda_graph_seq_cutoff", 64)
        previous = self.state.__dict__.copy()
        try:
            if graph and self.state.graph_runner is None:
                _enable_graphs(self.binding.weight_source, self.state, graph, cutoff)
            if train and self.state.train_graph_runner is None:
                _enable_train_graphs(self.binding.weight_source, self.state, train, cutoff)
        except BaseException:
            self.state.__dict__.update(previous)
            raise

    def set_cuda_graph(self, enabled: bool, config=None) -> None:
        from packed_encoders.graph import _set_modernbert_cuda_graph

        self._require_open()
        _set_modernbert_cuda_graph(self.binding.patch_target, enabled, config=config)

    def set_train_cuda_graph(self, enabled: bool, config=None) -> None:
        from packed_encoders.train_graph import _set_modernbert_train_cuda_graph

        self._require_open()
        _set_modernbert_train_cuda_graph(self.binding.patch_target, enabled, config=config)

    def close(self, *, rollback: bool = False) -> None:
        # ModernBERT never replaces or prepares copies of parameter storage.
        self._close_epilogues()
        self.state.graph_runner = None
        self.state.packed_graph_runner = None
        self.state.train_graph_runner = None
        self.state.graph_enabled = False
        self.state.train_graph_enabled = False
        self._closed = True
