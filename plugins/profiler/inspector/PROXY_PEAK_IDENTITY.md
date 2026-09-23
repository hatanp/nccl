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

## Family classification limits and compatibility

Family labels use configured topology cardinalities and the native operation
kind, not communicator membership or application call-site metadata. Non-global
AllReduce is now `unknown`: a size match cannot distinguish DP/EDP synchronization
from embedding or other model-parallel reductions. In particular, EDP2 can match
a two-endpoint embedding group even with PP4, and EDP4 can collide with PP4.
The plugin does not infer an `embedding` family. A world-sized collective remains
`global`, describing its extent rather than its application purpose.

AllGather/ReduceScatter retain the existing DP/EDP/EP size heuristics (priority:
world, DP, EDP, EP). These labels are useful with independently verified workload
geometry, but are not membership proof. P2P retains `pp_local`/`pp_cross_node`
according to node count; detached PXN retains `pxn` and unknown identity.

Prom format minor 8 (`v5.8` with profiler interface 5) marks this semantic
correction. Step records add a JSON `size_family_hint` field identifying their unchanged
legacy aggregation bin. It is a cardinality hint, not a semantic family claim.
`family` is the conservative semantic label. Consumers must key step rows by
step, operation **and size_family_hint** to preserve bin identity: several rows
can now have `family=unknown` for the same step and operation. For old captures,
the old `family` value identifies the legacy bin. Do not interpret a hint as
proof of DP/EDP membership or silently relabel unknown traffic as replica traffic.
Real replica AllReduce also becomes unknown until stronger evidence is available.

Step bins, event eligibility, counts, sums, maxima, GPU unions and interval
emission remain unchanged. Formerly DP/EDP bins still emit their merged GPU
intervals, including when their semantic family becomes unknown; formerly EP
bins still do not emit interval lists. No bins are merged and no operation
records are added. Previously unclassified step operations remain omitted.
Proxy aggregation retains communicator identity and emits unknown as before;
its schema is unchanged. No probes, communication, synchronization or profiling
enablement are added. Old captures keep their original labels and require
independent interpretation.

The portable `test-prom-stats` checks the EDP2 endpoint ambiguity, EDP4/PP4
collision, DP/EP ambiguity, global extent, preserved AG/RS/P2P heuristics,
legacy-bin separation and unchanged interval-emission policy.
These tests do not qualify native CUDA plugin compilation or emitted runtime
records; that qualification remains separate.

The native lifecycle fixture also checks a fixed DP-sized AllReduce descriptor
emits `family=unknown`, `size_family_hint=dp`, and its original timing fields and
fixed GPU interval, plus the current `v5.9` version label. Adding this fixture does not
constitute native execution or qualification on a host without CUDA.

## Optional collective parent metadata (Prom format v5.9)

Set `NCCL_INSPECTOR_PARENT_IDENTITY_ENABLE=1` in addition to proxy-step and
proxy-peak enablement. It is off by default and cannot activate either parent
feature by itself. This captures raw v5 collective descriptor values; it does
not add a matcher, profiler event class, communication or synchronization.
Existing aggregate keys, counts, sums, maxima, step eligibility and the original
witness tie rule are unchanged. Parent IDs do not split aggregates.

Each successful local collective allocation gets a monotonically increasing
64-bit lifetime ID. Zero means unavailable; the counter saturates instead of
wrapping. It is not reset by communicator init/finalize. A 128-bit random hex
`parent_process_instance` is generated once per plugin load, during instrument
initialization, from `/dev/urandom`. Failure leaves metadata unavailable. Pair
namespace and ID when combining independently saved outputs. This is a local
plugin-instance identity with probabilistic namespace uniqueness, not a global
application/graph identifier. Forking an initialized plugin is not qualified.

`# nccl_inspector_proxy_parent ` records use schema 1 and contain:

- `process_instance` and `parent_id`;
- `send_buffer` and `recv_buffer` as hexadecimal address strings;
- `native_count` and `native_datatype` in canonical NCCL spelling.

Only parents referenced by retained maxima enter the dump-time dictionary.
Each usable parent is emitted once per self-contained output snapshot, even
when multiple phases reference it. Existing schema-1 peak rows add `parent_id`
and `parent_metadata_known`; their existing `identity_known` still describes
local versus detached originating identity, not join qualification. Dictionary
rows inherit communicator/function context from referencing peaks.

Proxy-info rows add `parent_identity_schema`, `parent_identity_enabled`,
`parent_process_instance`, `parent_dictionary_capacity`,
`parent_dictionary_entries` (usable emitted entries), and
`parent_metadata_unavailable_peak_records` (local collective peak references
without usable retained metadata, **not** unique missing parents). P2P and
foreign detached PXN have parent ID zero and no collective dictionary metadata.
The foreign origin PID guard remains before any parent/context dereference.

`NCCL_INSPECTOR_PARENT_DICTIONARY_CAPACITY` defaults to 4096 and permits 0–65536;
zero and invalid/oversized values fail closed to no dictionary retention.
The lowest K parent IDs are retained independent of traversal order. Invalid
metadata occupies its ID slot but emits no usable row; conflicting copies of
one ID also fail closed. Saturation, missing/null buffers, zero count, unknown
native datatype or absent namespace never rebind an ID or discard a numeric
peak. Consumers requiring parent data must reject the resulting missing
coverage rather than guessing a match.

Callback storage is fixed: 40 bytes per allocated collective parent, eight
bytes of parent ID per proxy op and per phase witness on the measured LP64
layout. Full metadata is **not** duplicated in each proxy op or step. The live
local parent is already reference-held through proxy-op finalization; at that
point the selected metadata is copied into the retained aggregate alongside
its witness before the parent reference is released. Each aggregate adds six
40-byte snapshots plus six eight-byte witness IDs (288 bytes). Default 8192-op
pool growth is 458752 bytes; the proxy-step object is unchanged. These increases
exist in the compiled structs even when capture is disabled; opt-in controls
ID allocation, copying and dictionary emission.

The dump-time ordered map holds at most K entries, each with a 48-byte payload
plus key/tree/allocator overhead on the measured host. No unbounded all-event
registry is added. Added storage remains proportional to existing allocated
parents/aggregates, whose original pools/maps are not universally hard-capped;
this is not a new absolute cap on the plugin's baseline memory. At most
min(K, retained collective witness count) dictionary rows are added, with no
increase in peak-record count. Namespace setup performs cold-path file I/O;
per-transfer callbacks perform no new I/O, allocation or dictionary lookup.
Overhead is not assumed negligible and needs separate native measurement.

Raw descriptors are not automatically equal to PyTorch NVTX arguments. The
inspected local NCCL source lowers AllGather count to bytes and datatype to
`ncclInt8`; the exact loaded runtime lowering still needs qualification.
ReduceScatter's reduction operator is absent from v5. Buffer addresses may
repeat across operations/replays, and a grouped launch can contain several
tensor calls. IDs distinguish parent lifetimes but do not supply an NVTX
capture-launch ID, a replay identity, an ordinal join or cross-clock causality.
Zero/multiple qualified candidates remain unknown.

`test-parent-identity` is portable: value copying, strict unknown handling,
concurrent/saturating IDs, stable namespace, witness replacement/ties, immutable
metadata pairing, deterministic caps/conflicts and struct sizes. The native
`test-proxy-lifecycle` fixture exercises same-sequence/different-buffer parents,
a task stopped before its proxy child, address reuse with a new lifetime ID,
mutable callback descriptors, valid/no/saturated dictionaries, missing buffers,
and invalid foreign parent/context pointers. Its native runs are not available
on the Mac; syntax checking is not execution qualification.

GPU graph-replay qualification is explicitly deferred: replay a fixed captured
AG/RS group at least twice with distinct and reused buffers, observe parent-ID
lifetimes and callback count/datatype lowering, and verify marker assignment
and retained dictionary references. Neither portable tests nor synthetic host
callbacks establish that behavior in image 68817788. Preserve old captures and
release identities; no retrofit of their missing buffer metadata is possible.
