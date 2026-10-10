# Deferred DP-aware PD integration

Status: **deferred**, 2026-10-10. This branch preserves unfinished integration
for review and future resumption; it is not a release or merge proposal.

The work depends on downstream gateway behavior that has not shipped in the
published packages reviewed for this integration. Remaining lifecycle and rank
contracts need further review. Keep the existing endpoint-level PD behavior on
active branches. Bug fixes already merged into main, including PD port allocation,
are outside this archive commit and must remain in main.

## Preserved implementation

Optional SGLang request-rank routing for the managed Mooncake/NIXL recipes:
effective per-role DP/DPA parsing, physical URL versus logical rank reconciliation,
whole-group deletion/read-back, patched-artifact checking, independent P/D policies,
and actual-prefill-rank bootstrap configuration. vLLM IGW rank registration and
external/hybrid deployment integration are not implemented here.

Historical validation: 291 focused GPUStack tests, 315 runner packaging tests,
3 Rust contract tests, and real compiled gateway/mock-engine P2/D4 and P1/D2
runs (74 requests each) passed. These establish CPU contracts only. No full CUDA
image was built/published; no GPUStack runtime upgrade or real SGLang GPU KV
transfer was validated.

## Dependencies and resumption

The companion runner branch pins SGLang gateway source
`f620d733d256c535b00d6e2f6b8e43c27fa81214` plus a local patch and labels the wheel
`0.3.2+gpustack.dp1`; the engine remains 0.5.18. Do not enable this integration
with the reviewed unpatched public 0.3.2 artifact.

Before resuming, compare released gateway APIs and PD dispatch to the preserved
contracts. Revisit parent registration IDs versus rank IDs, whole-group versus
single-rank removal, client-rank precedence, and old/new DP/DPA layouts. Prefer
released upstream behavior where it meets the contract; reassess local patches
and the version marker rather than carrying them forward automatically. Then
rebase onto current main, run regressions, build the final image and validate
actual rank execution, KV transfer, membership recovery, and routing-key forwarding.

Companion branches:

- [runner](https://github.com/aiwantaozi/runner/tree/archive/dp-aware-20261010)
- [plans and step-by-step validation](https://github.com/aiwantaozi/plans/tree/archive/dp-aware-20261010/gpustack/pd-disaggregation/dp)

To inspect without enabling it in an active checkout, fetch this branch and
create a separate worktree. This snapshot starts at GPUStack
`102ec612f8d8ebf4d2f22e5de3890fbbae8cb3ed`; it is deliberately not rebased for
archiving. Use only the archive commit when transplanting the change; do not
reapply already merged bug fixes.
