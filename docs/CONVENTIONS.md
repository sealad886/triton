# Repository Conventions

## Scope and Purpose

This document captures non-obvious repository conventions that affect how core
subsystems should evolve. It is intentionally short and focuses on patterns a
new contributor could reasonably miss while still writing otherwise-correct
code.

## Core Conventions

### Metal capability truth source

- **Status**: REQUIRED
- **Scope**: Metal backend compiler/runtime, CI probes, tests, and user-facing Metal docs
- **Rule**: Source Metal feature-support claims and launch/runtime capability checks from `python/triton/backends/metal/capabilities.py` (or its `third_party` mirror) instead of duplicating support tables in code.

## Rationale and Examples

- The Metal backend intentionally has multiple support levels: fully supported,
  limited/metadata-only, and unsupported. Encoding those decisions in one
  machine-readable module keeps compiler checks, runtime probes, CI reporting,
  and docs aligned.
- Example: `launch_pdl` and `profile_scratch` are not "fully supported"
  features. They are represented as limited capabilities in the shared snapshot,
  and docs/CI should report them that way rather than as blanket ✅ support.

## Known Exceptions

- Static release notes or audit documents may summarize capability state in
  prose for a point-in-time snapshot, but they should still reflect the shared
  capability snapshot rather than inventing independent status language.

## Change History

- 2026-03-14: Added the Metal capability truth-source convention after
  centralizing launch/runtime/parity status in `python/triton/backends/metal/capabilities.py`.
