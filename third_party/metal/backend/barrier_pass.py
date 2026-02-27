"""
Dedicated Metal shared-memory barrier/fence insertion pass.

Replaces the ad-hoc translator-level heuristic for barrier placement
with a structured, pattern-aware pass that analyzes shared memory
access patterns and inserts Metal threadgroup barriers at required
synchronization points.

Operates on textual LLVM IR before MSL translation. The pass:
  1. Parses the IR into basic blocks per function.
  2. Identifies loads/stores to addrspace(3) (threadgroup/shared memory).
  3. Detects producer→consumer pairs needing a barrier.
  4. Handles loop-carried dependencies via backedge detection.
  5. Inserts ``call void @llvm.nvvm.barrier0()`` at synchronization points.
  6. Is idempotent: a second run produces identical output.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_BARRIER_DEBUG = os.environ.get("TRITON_METAL_BARRIER_DEBUG", "").lower() in (
    "1",
    "true",
    "yes",
)

# ── Shared-memory access detection regexes ──────────────────────────

_RE_STORE_AS3 = re.compile(
    r"store\s+.+?,\s+(?:ptr|.*\*)\s+addrspace\(3\)"
    r"|"
    r"store\s+.+?,\s+ptr\s+addrspace\(3\)"
)
_RE_LOAD_AS3 = re.compile(
    r"=\s*load\s+.+?,\s+(?:ptr|.*\*)\s+addrspace\(3\)"
    r"|"
    r"=\s*load\s+.+?,\s+ptr\s+addrspace\(3\)"
)
_RE_GLOBAL_SMEM = re.compile(r"@global_smem\b")
_RE_ADDRSPACE3_GEP = re.compile(r"getelementptr\s+.*addrspace\(3\)")

_RE_EXISTING_BARRIER = re.compile(
    r"call\s+void\s+@llvm\.nvvm\.barrier0\(\)"
    r"|"
    r"fence\s+syncscope\(\"workgroup\"\)\s+seq_cst"
)

_RE_LABEL = re.compile(r"^([A-Za-z0-9_.]+):\s*(?:;.*)?$")
_RE_BRANCH = re.compile(r"br\s+label\s+%([A-Za-z0-9_.]+)")
_RE_COND_BRANCH = re.compile(
    r"br\s+i1\s+.+?,\s+label\s+%([A-Za-z0-9_.]+),\s+label\s+%([A-Za-z0-9_.]+)"
)
_RE_FUNC_START = re.compile(r"^define\s+")
_RE_FUNC_END = re.compile(r"^}$")

_BARRIER_INTRINSIC = "  call void @llvm.nvvm.barrier0()"


@dataclass
class BarrierDecision:
    """Records why and where a barrier was inserted."""

    block_label: str
    insertion_index: int
    reason: str


@dataclass
class BasicBlock:
    """A parsed basic block from LLVM IR."""

    label: str
    lines: list[str] = field(default_factory=list)
    has_shared_store: bool = False
    has_shared_load: bool = False
    successor_labels: list[str] = field(default_factory=list)


def _line_is_shared_store(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("store"):
        return False
    return "addrspace(3)" in stripped or "@global_smem" in stripped


def _line_is_shared_load(line: str) -> bool:
    stripped = line.strip()
    if "= load" not in stripped:
        return False
    return "addrspace(3)" in stripped or "@global_smem" in stripped


def _line_is_shared_gep(line: str) -> bool:
    stripped = line.strip()
    return "getelementptr" in stripped and (
        "addrspace(3)" in stripped or "@global_smem" in stripped
    )


def _line_is_barrier(line: str) -> bool:
    stripped = line.strip()
    return (
        "call void @llvm.nvvm.barrier0()" in stripped
        or 'fence syncscope("workgroup") seq_cst' in stripped
    )


def _parse_function_blocks(func_lines: list[str]) -> list[BasicBlock]:
    """Parse LLVM IR function body into BasicBlocks."""
    blocks: list[BasicBlock] = []
    current_block: BasicBlock | None = None

    for line in func_lines:
        stripped = line.strip()

        label_m = _RE_LABEL.match(stripped)
        if label_m:
            if current_block is not None:
                blocks.append(current_block)
            current_block = BasicBlock(label=label_m.group(1))
            continue

        if current_block is None:
            current_block = BasicBlock(label="__entry__")

        current_block.lines.append(line)

        if _line_is_shared_store(line):
            current_block.has_shared_store = True
        if _line_is_shared_load(line):
            current_block.has_shared_load = True

        cond_m = _RE_COND_BRANCH.search(stripped)
        if cond_m:
            current_block.successor_labels.extend([cond_m.group(1), cond_m.group(2)])
        else:
            br_m = _RE_BRANCH.search(stripped)
            if br_m:
                current_block.successor_labels.append(br_m.group(1))

    if current_block is not None:
        blocks.append(current_block)

    return blocks


def _has_backedge(blocks: list[BasicBlock]) -> bool:
    """Detect if any branch targets an earlier block (loop backedge)."""
    label_order: dict[str, int] = {}
    for i, blk in enumerate(blocks):
        label_order[blk.label] = i

    for i, blk in enumerate(blocks):
        for succ in blk.successor_labels:
            target_idx = label_order.get(succ)
            if target_idx is not None and target_idx <= i:
                return True
    return False


def _find_loop_header_blocks(blocks: list[BasicBlock]) -> set[str]:
    """Find block labels that are targets of backedges (loop headers)."""
    label_order: dict[str, int] = {}
    for i, blk in enumerate(blocks):
        label_order[blk.label] = i

    loop_headers: set[str] = set()
    for i, blk in enumerate(blocks):
        for succ in blk.successor_labels:
            target_idx = label_order.get(succ)
            if target_idx is not None and target_idx <= i:
                loop_headers.add(succ)
    return loop_headers


class MetalBarrierInsertionPass:
    """
    Analyzes LLVM IR for shared-memory access patterns and inserts
    barrier intrinsics at required synchronization points.

    Usage::

        pass_ = MetalBarrierInsertionPass()
        new_ir = pass_.run(llvm_ir_text)
        decisions = pass_.decisions  # list[BarrierDecision]
    """

    def __init__(self, *, debug: bool | None = None) -> None:
        self.debug = debug if debug is not None else _BARRIER_DEBUG
        self.decisions: list[BarrierDecision] = []

    def run(self, llvm_ir: str) -> str:
        """
        Analyze *llvm_ir* and insert barriers where needed.

        Returns the modified IR string. Populates ``self.decisions``.
        """
        self.decisions.clear()

        if "@global_smem" not in llvm_ir and "addrspace(3)" not in llvm_ir:
            if self.debug:
                print("[BARRIER_PASS] No shared memory references found, skipping.")
            return llvm_ir

        lines = llvm_ir.split("\n")
        result_lines: list[str] = []
        in_function = False
        func_lines: list[str] = []
        func_start_line: str = ""
        brace_depth = 0

        for line in lines:
            stripped = line.strip()

            if not in_function and _RE_FUNC_START.match(stripped):
                in_function = True
                func_lines = []
                func_start_line = line
                brace_depth = stripped.count("{") - stripped.count("}")
                if brace_depth > 0:
                    result_lines.append(line)
                    continue
                else:
                    result_lines.append(line)
                    continue

            if in_function:
                brace_depth += stripped.count("{") - stripped.count("}")
                if brace_depth <= 0:
                    processed = self._process_function(func_lines)
                    result_lines.extend(processed)
                    result_lines.append(line)
                    in_function = False
                    func_lines = []
                    continue
                func_lines.append(line)
                continue

            result_lines.append(line)

        if self.debug and self.decisions:
            print(f"[BARRIER_PASS] Inserted {len(self.decisions)} barrier(s):")
            for d in self.decisions:
                print(f"  Block '{d.block_label}': {d.reason}")

        return "\n".join(result_lines)

    def _process_function(self, func_lines: list[str]) -> list[str]:
        """Process a single function's body and insert barriers as needed."""
        blocks = _parse_function_blocks(func_lines)

        if not blocks:
            return func_lines

        any_shared_store = any(b.has_shared_store for b in blocks)
        any_shared_load = any(b.has_shared_load for b in blocks)

        if not any_shared_store or not any_shared_load:
            return func_lines

        has_loop = _has_backedge(blocks)
        loop_headers = _find_loop_header_blocks(blocks) if has_loop else set()

        label_to_block: dict[str, BasicBlock] = {b.label: b for b in blocks}
        store_blocks = {b.label for b in blocks if b.has_shared_store}

        new_func_lines: list[str] = []

        for blk in blocks:
            if blk.label != "__entry__":
                new_func_lines.append(f"{blk.label}:")

            blk_lines = self._insert_barriers_in_block(
                blk,
                store_blocks=store_blocks,
                loop_headers=loop_headers,
                label_to_block=label_to_block,
                has_loop=has_loop,
            )
            new_func_lines.extend(blk_lines)

        return new_func_lines

    def _insert_barriers_in_block(
        self,
        block: BasicBlock,
        *,
        store_blocks: set[str],
        loop_headers: set[str],
        label_to_block: dict[str, BasicBlock],
        has_loop: bool,
    ) -> list[str]:
        """
        Insert barriers within a single block where needed.

        Strategy (conservative):
        1. If the block loads from shared memory AND some block stores to
           shared memory AND we're in a loop, insert a barrier before the
           first shared load (unless one already exists before it).
        2. For producer→consumer within a single block (store followed by
           load), insert a barrier between them.
        3. Before a backedge branch in a block that stores to shared memory,
           insert a barrier so the next loop iteration sees the stores.
        """
        result: list[str] = []
        barrier_already_present = False
        seen_shared_store_in_block = False

        for i, line in enumerate(block.lines):
            if _line_is_barrier(line):
                barrier_already_present = True
                result.append(line)
                continue

            is_store = _line_is_shared_store(line)
            is_load = _line_is_shared_load(line)

            if is_load:
                needs_barrier = False
                reason = ""

                if seen_shared_store_in_block and not barrier_already_present:
                    needs_barrier = True
                    reason = "producer→consumer within block (store before load)"

                elif (
                    has_loop
                    and store_blocks
                    and not barrier_already_present
                    and block.label in loop_headers
                ):
                    needs_barrier = True
                    reason = "loop header with shared memory stores in loop body"

                elif (
                    has_loop
                    and store_blocks
                    and not barrier_already_present
                    and block.has_shared_store
                ):
                    needs_barrier = True
                    reason = "shared load in loop block with stores"

                if needs_barrier:
                    result.append(_BARRIER_INTRINSIC)
                    self.decisions.append(
                        BarrierDecision(
                            block_label=block.label,
                            insertion_index=len(result) - 1,
                            reason=reason,
                        )
                    )
                    barrier_already_present = True

            if is_store:
                seen_shared_store_in_block = True
                barrier_already_present = False

            stripped = line.strip()
            is_backedge_branch = False
            if stripped.startswith("br "):
                cond_m = _RE_COND_BRANCH.search(stripped)
                if cond_m:
                    for target_label in [cond_m.group(1), cond_m.group(2)]:
                        if target_label in loop_headers:
                            is_backedge_branch = True
                            break
                else:
                    br_m = _RE_BRANCH.search(stripped)
                    if br_m and br_m.group(1) in loop_headers:
                        is_backedge_branch = True

            if (
                is_backedge_branch
                and (block.has_shared_store or store_blocks)
                and not barrier_already_present
            ):
                result.append(_BARRIER_INTRINSIC)
                self.decisions.append(
                    BarrierDecision(
                        block_label=block.label,
                        insertion_index=len(result) - 1,
                        reason="barrier before backedge to protect shared stores",
                    )
                )
                barrier_already_present = True

            result.append(line)

        return result


def ensure_barrier_declaration(llvm_ir: str) -> str:
    """
    Ensure the ``@llvm.nvvm.barrier0`` declaration exists in the IR
    module if any barrier calls were inserted.
    """
    if "call void @llvm.nvvm.barrier0()" not in llvm_ir:
        return llvm_ir
    if "declare void @llvm.nvvm.barrier0()" in llvm_ir:
        return llvm_ir
    lines = llvm_ir.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("define "):
            insert_idx = i
            break
    lines.insert(insert_idx, "declare void @llvm.nvvm.barrier0()")
    if insert_idx > 0 and lines[insert_idx - 1].strip():
        lines.insert(insert_idx, "")
    return "\n".join(lines)


def run_barrier_pass(llvm_ir: str, *, debug: bool | None = None) -> str:
    """
    Convenience entry point: run the barrier insertion pass and return
    the modified LLVM IR.
    """
    pass_ = MetalBarrierInsertionPass(debug=debug)
    result = pass_.run(llvm_ir)
    result = ensure_barrier_declaration(result)
    return result
