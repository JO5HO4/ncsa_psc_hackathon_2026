# Native agent tool contract (v1)

We use **native Codex and OpenCode tools only**. No MCP server or custom tool
wrapper is part of the dataset. The canonical source task declares capabilities
(`read`, `modify`, and `execute`); each harness renderer emits its native tool
calls and records their native results.

The two renderings are equivalent task solutions, not byte-identical tool
traces. They share a `logical_task_id`, fixture, expected final state, verifier
result, and split.

## Allowed native tools

| Capability | Codex rendering | OpenCode rendering |
| --- | --- | --- |
| Read or search workspace files | `Bash` running bounded commands such as `sed`, `rg`, `head`, or `tail` | `read`; `grep`, `glob`, or `list` where needed |
| Modify a tracked config | `apply_patch` with a unified diff | `edit`, `write`, or `apply_patch` |
| Validate, run, or inspect local artifacts | `Bash` | `bash` |

For Codex, file reads are deliberately represented as `Bash` calls. Prefer
line-bounded reads such as `sed -n '1,160p' configs/task.config` and bounded
searches such as `rg -n 'Region|Sample' configs/task.config`; do not emit an
unbounded file dump. `apply_patch` is the only file mutation tool in Codex
records.

For OpenCode, prefer its native `read` tool for file contents, `edit` for an
exact replacement, `write` only when complete replacement is intended, and
`apply_patch` for a unified diff. The OpenCode rendering may instead use `bash`
for local validators and runners.

## Sandbox and authoring rules

- Every command runs in the task's private fixture-derived workspace. The
  harness configuration fixes the working directory, resource limits, output
  limit, and maximum timeout.
- Dataset task environments must disable network access and provide no
  credentials. The model never supplies a working directory, timeout above the
  task cap, or environment secrets.
- Paths must be workspace-relative and restricted to the task's allowlisted
  files. Patches may change only allowlisted paths.
- Record observable native tool output verbatim, including exit status and
  truncated stdout/stderr. Do not synthesize output or expose hidden verifier
  state.
- A task advertises only the native tools it needs. A config-repair task
  normally enables `Bash` and `apply_patch` for Codex, and `read`, `edit` or
  `apply_patch`, and `bash` for OpenCode.
- Direct-config rows have no tool manifest (`tools: []`).

## Renderer requirements

The renderer owns the native tool names, parameter schemas, and tool-result
serialization. The source task must not contain Codex or OpenCode wrapper
tokens. Before release, replay each rendering in its target harness against the
same fixture and require the same verified final config state.
