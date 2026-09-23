# Identity-preserving proxy phase maxima

Set `NCCL_INSPECTOR_PROXY_STEP_ENABLE=1` and
`NCCL_INSPECTOR_PROXY_PEAK_ENABLE=1` alongside the existing step/Prometheus
settings. Peak output is opt-in. Existing aggregate counts, sums and maxima
keep their definitions. Prom format minor 7 adds the peak capability fields.

Each existing step/family/function/size/direction/communicator aggregate retains
one exact witness for each phase maximum. Selection copies the entire witness;
its identity fields are never independently reduced. Equal maxima use a stable
tie rule. The hot path stores fixed-size records in the existing operation pool
and adds no file I/O or allocation. Six witness slots add
`6 * sizeof(inspectorProxyPhasePeak)` bytes per pooled operation and aggregate,
including when peak output is disabled. Output is bounded by six records per
existing aggregate; there is no per-channel or per-sequence expansion of keys.

New comment records have prefix `# nccl_inspector_step_proxy_peak ` and JSON
schema 1. They retain application step, family, function, message bytes,
direction, communicator hash/rank/size, native parent type and sequence,
channel, raw proxy peer, transfer step, phase start/stop/duration, and
`identity_known`. Timestamps use the host `gettimeofday` clock, not the rejected
GPU capture-to-replay pTimer path. Backward/zero-start intervals do not qualify.

Limits that consumers must retain:

- This is the exact event behind a local maximum, not an exhaustive event log.
  A missing peer counterpart is unknown, not evidence it did not happen.
- Detached PXN events never dereference a foreign parent. Their originating
  identity remains unknown (`identity_known=false`); sequence/parent placeholders
  cannot be used as a communicator join.
- NCCL collective sequence is per communicator/function and originates at task
  construction. Its reuse under CUDA-graph replay must be qualified. The
  application-step marker and host timestamp do not independently prove a
  unique application-operation mapping.
- Proxy peer is retained in the native descriptor's rank space. Qualify that
  mapping, especially with PXN, before equating it to a global rank.
- Host clocks are explicitly marked unaligned. Do not infer cross-host ordering
  from absolute timestamps without a clock calibration/error bound.
- Waiting for GPU production/consumption describes a local state, not its
  upstream cause or a fraction of global training critical path.

Validation: the portable `test-proxy-stats` target checks selection, ties,
invalid intervals and witness integrity. `test-proxy-lifecycle` checks emitted
metadata, native parent lifetime and unknown detached identity. The latter
requires a native CUDA build; portable helper tests do not qualify the plugin
or CUDA-graph replay. Reuse the exact workload image for native qualification.
