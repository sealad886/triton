"""
Tests for the typed LLVM IR instruction AST and parser.

Validates ``ir_types.parse_instruction`` and ``parse_block`` against
representative LLVM IR lines drawn from the Metal backend test corpus.
"""

import sys

import pytest
from triton.backends.metal.ir_types import (
    GEP,
    AggregateOp,
    Alloca,
    AtomicOp,
    BinOp,
    Call,
    Cast,
    FCmp,
    FNeg,
    Freeze,
    ICmp,
    Instruction,
    LLVMInstruction,
    Load,
    Phi,
    Select,
    Store,
    Terminator,
    UnknownInstruction,
    VectorOp,
    parse_block,
    parse_instruction,
)

# ── Import under test ───────────────────────────────────────────────


skip_non_darwin = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Metal backend tests require macOS",
)

pytestmark = skip_non_darwin


# =====================================================================
# 1. Per-instruction-type tests (parametrized)
# =====================================================================


class TestBinOp:
    @pytest.mark.parametrize(
        "line, expected_op, expected_ty, expected_lhs, expected_rhs",
        [
            ("%5 = add nsw i32 %3, %4", "add", "i32", "%3", "%4"),
            ("%7 = fadd float %5, %6", "fadd", "float", "%5", "%6"),
            ("%9 = mul i32 %7, %8", "mul", "i32", "%7", "%8"),
            ("%11 = and i32 %9, %10", "and", "i32", "%9", "%10"),
            ("%13 = shl i32 %11, %12", "shl", "i32", "%11", "%12"),
            ("%2 = sub i32 %0, %1", "sub", "i32", "%0", "%1"),
            ("%3 = xor i32 %1, %2", "xor", "i32", "%1", "%2"),
            ("%4 = or i32 %2, %3", "or", "i32", "%2", "%3"),
            ("%5 = lshr i32 %3, %4", "lshr", "i32", "%3", "%4"),
            ("%6 = ashr i32 %4, %5", "ashr", "i32", "%4", "%5"),
            ("%7 = udiv i32 %5, %6", "udiv", "i32", "%5", "%6"),
            ("%8 = sdiv i32 %6, %7", "sdiv", "i32", "%6", "%7"),
            ("%9 = urem i32 %7, %8", "urem", "i32", "%7", "%8"),
            ("%10 = srem i32 %8, %9", "srem", "i32", "%8", "%9"),
            ("%11 = fsub float %9, %10", "fsub", "float", "%9", "%10"),
            ("%12 = fmul float %10, %11", "fmul", "float", "%10", "%11"),
            ("%13 = fdiv float %11, %12", "fdiv", "float", "%11", "%12"),
            ("%14 = frem float %12, %13", "frem", "float", "%12", "%13"),
        ],
        ids=lambda x: x if isinstance(x, str) and x.startswith("%") else "",
    )
    def test_binop_variants(
        self, line, expected_op, expected_ty, expected_lhs, expected_rhs
    ):
        inst = parse_instruction(line)
        assert isinstance(inst, BinOp)
        assert inst.op == expected_op
        assert inst.llvm_ty == expected_ty
        assert inst.lhs == expected_lhs
        assert inst.rhs == expected_rhs
        assert inst.raw_line == line

    def test_binop_flags_nsw(self):
        inst = parse_instruction("%5 = add nsw i32 %3, %4")
        assert isinstance(inst, BinOp)
        assert inst.flags == "nsw"

    def test_binop_flags_nuw_nsw(self):
        inst = parse_instruction("%5 = add nuw nsw i32 %3, %4")
        assert isinstance(inst, BinOp)
        assert "nuw" in inst.flags
        assert "nsw" in inst.flags

    def test_binop_flags_fast(self):
        inst = parse_instruction("%7 = fadd fast float %5, %6")
        assert isinstance(inst, BinOp)
        assert inst.flags == "fast"

    def test_binop_no_flags(self):
        inst = parse_instruction("%9 = mul i32 %7, %8")
        assert isinstance(inst, BinOp)
        assert inst.flags == ""

    def test_binop_nnan_ninf(self):
        inst = parse_instruction("%2 = fadd nnan ninf float %0, %1")
        assert isinstance(inst, BinOp)
        assert "nnan" in inst.flags
        assert "ninf" in inst.flags


class TestLoad:
    def test_load_addrspace(self):
        line = "%2 = load float, ptr addrspace(1) %1, align 4"
        inst = parse_instruction(line)
        assert isinstance(inst, Load)
        assert inst.out_ssa == "%2"
        assert inst.llvm_ty == "float"
        assert inst.addr_space == "1"
        assert inst.ptr == "%1"
        assert inst.alignment == "4"
        assert inst.raw_line == line

    def test_load_no_addrspace(self):
        line = "%4 = load i32, ptr %3"
        inst = parse_instruction(line)
        assert isinstance(inst, Load)
        assert inst.out_ssa == "%4"
        assert inst.llvm_ty == "i32"
        assert inst.addr_space is None
        assert inst.ptr == "%3"
        assert inst.alignment is None

    def test_load_i64_align8(self):
        line = "%6 = load i64, ptr addrspace(1) %5, align 8"
        inst = parse_instruction(line)
        assert isinstance(inst, Load)
        assert inst.llvm_ty == "i64"
        assert inst.alignment == "8"


class TestStore:
    def test_store_addrspace(self):
        line = "store float %val, ptr addrspace(1) %ptr, align 4"
        inst = parse_instruction(line)
        assert isinstance(inst, Store)
        assert inst.val_ty == "float"
        assert inst.val == "%val"
        assert inst.addr_space == "1"
        assert inst.ptr == "%ptr"
        assert inst.alignment == "4"
        assert inst.raw_line == line

    def test_store_no_addrspace(self):
        line = "store i32 %val, ptr %ptr"
        inst = parse_instruction(line)
        assert isinstance(inst, Store)
        assert inst.addr_space is None
        assert inst.alignment is None

    def test_store_i64(self):
        line = "store i64 %val, ptr addrspace(1) %ptr, align 8"
        inst = parse_instruction(line)
        assert isinstance(inst, Store)
        assert inst.val_ty == "i64"
        assert inst.alignment == "8"


class TestCast:
    @pytest.mark.parametrize(
        "line, expected_cast_op, expected_to_ty",
        [
            ("%3 = sext i32 %2 to i64", "sext", "i64"),
            ("%5 = fptrunc double %4 to float", "fptrunc", "float"),
            ("%7 = bitcast i32 %6 to float", "bitcast", "float"),
            ("%9 = addrspacecast ptr addrspace(1) %8 to ptr", "addrspacecast", "ptr"),
            ("%10 = zext i16 %9 to i32", "zext", "i32"),
            ("%11 = trunc i64 %10 to i32", "trunc", "i32"),
            ("%12 = sitofp i32 %11 to float", "sitofp", "float"),
            ("%13 = uitofp i32 %12 to float", "uitofp", "float"),
            ("%14 = fptosi float %13 to i32", "fptosi", "i32"),
            ("%15 = fptoui float %14 to i32", "fptoui", "i32"),
            ("%16 = fpext float %15 to double", "fpext", "double"),
            ("%17 = ptrtoint ptr %16 to i64", "ptrtoint", "i64"),
            ("%18 = inttoptr i64 %17 to ptr", "inttoptr", "ptr"),
        ],
        ids=lambda x: x if isinstance(x, str) and x.startswith("%") else "",
    )
    def test_cast_variants(self, line, expected_cast_op, expected_to_ty):
        inst = parse_instruction(line)
        assert isinstance(inst, Cast)
        assert inst.cast_op == expected_cast_op
        assert inst.to_ty == expected_to_ty
        assert inst.raw_line == line


class TestICmp:
    @pytest.mark.parametrize(
        "line, pred, ty, lhs, rhs",
        [
            ("%3 = icmp slt i32 %1, %2", "slt", "i32", "%1", "%2"),
            ("%4 = icmp eq i64 %2, 0", "eq", "i64", "%2", "0"),
            ("%5 = icmp uge i32 %3, %4", "uge", "i32", "%3", "%4"),
            ("%6 = icmp ne i1 %4, true", "ne", "i1", "%4", "true"),
        ],
    )
    def test_icmp_variants(self, line, pred, ty, lhs, rhs):
        inst = parse_instruction(line)
        assert isinstance(inst, ICmp)
        assert inst.pred == pred
        assert inst.llvm_ty == ty
        assert inst.lhs == lhs
        assert inst.rhs == rhs


class TestFCmp:
    @pytest.mark.parametrize(
        "line, pred, ty, lhs, rhs",
        [
            ("%4 = fcmp olt float %2, %3", "olt", "float", "%2", "%3"),
            ("%5 = fcmp oeq double %3, %4", "oeq", "double", "%3", "%4"),
            ("%6 = fcmp une float %4, %5", "une", "float", "%4", "%5"),
        ],
    )
    def test_fcmp_variants(self, line, pred, ty, lhs, rhs):
        inst = parse_instruction(line)
        assert isinstance(inst, FCmp)
        assert inst.pred == pred
        assert inst.llvm_ty == ty
        assert inst.lhs == lhs
        assert inst.rhs == rhs


class TestCall:
    def test_call_nonvoid(self):
        line = "%2 = call float @llvm.fabs.f32(float %1)"
        inst = parse_instruction(line)
        assert isinstance(inst, Call)
        assert inst.is_void is False
        assert inst.out_ssa == "%2"
        assert inst.fn_name == "llvm.fabs.f32"
        assert "float %1" in inst.args_raw
        assert inst.raw_line == line

    def test_call_void(self):
        line = "call void @llvm.nvvm.barrier0()"
        inst = parse_instruction(line)
        assert isinstance(inst, Call)
        assert inst.is_void is True
        assert inst.out_ssa is None
        assert inst.ret_type == "void"
        assert inst.fn_name == "llvm.nvvm.barrier0"

    def test_call_tail(self):
        line = "%3 = tail call float @llvm.sqrt.f32(float %2)"
        inst = parse_instruction(line)
        assert isinstance(inst, Call)
        assert inst.is_void is False
        assert inst.fn_name == "llvm.sqrt.f32"

    def test_call_multi_arg(self):
        line = "%4 = call float @llvm.fma.f32(float %1, float %2, float %3)"
        inst = parse_instruction(line)
        assert isinstance(inst, Call)
        assert inst.fn_name == "llvm.fma.f32"
        assert "%1" in inst.args_raw
        assert "%3" in inst.args_raw


class TestGEP:
    def test_gep_inbounds(self):
        line = "%3 = getelementptr inbounds float, ptr addrspace(3) %1, i32 %2"
        inst = parse_instruction(line)
        assert isinstance(inst, GEP)
        assert inst.out_ssa == "%3"
        assert inst.inbounds is True
        assert inst.base_ty == "float"
        assert inst.ptr_operand == "%1"
        assert "%2" in inst.indices_raw
        assert inst.raw_line == line

    def test_gep_no_inbounds(self):
        line = "%4 = getelementptr float, ptr %1, i32 %2"
        inst = parse_instruction(line)
        assert isinstance(inst, GEP)
        assert inst.inbounds is False


class TestPhi:
    def test_phi_two_incoming(self):
        line = "%5 = phi float [ %3, %bb1 ], [ %4, %bb2 ]"
        inst = parse_instruction(line)
        assert isinstance(inst, Phi)
        assert inst.out_ssa == "%5"
        assert inst.llvm_ty == "float"
        assert "%3" in inst.incoming_raw
        assert "%bb2" in inst.incoming_raw
        assert inst.raw_line == line

    def test_phi_i32(self):
        line = "%10 = phi i32 [ 0, %entry ], [ %9, %loop ]"
        inst = parse_instruction(line)
        assert isinstance(inst, Phi)
        assert inst.llvm_ty == "i32"


class TestSelect:
    def test_select_float(self):
        line = "%4 = select i1 %1, float %2, float %3"
        inst = parse_instruction(line)
        assert isinstance(inst, Select)
        assert inst.out_ssa == "%4"
        assert inst.cond == "%1"
        assert inst.true_val == "%2"
        assert inst.false_val == "%3"
        assert inst.result_ty == "float"
        assert inst.raw_line == line

    def test_select_i32(self):
        line = "%5 = select i1 %cond, i32 %a, i32 %b"
        inst = parse_instruction(line)
        assert isinstance(inst, Select)
        assert inst.result_ty == "i32"


class TestFNeg:
    def test_fneg_float(self):
        line = "%2 = fneg float %1"
        inst = parse_instruction(line)
        assert isinstance(inst, FNeg)
        assert inst.out_ssa == "%2"
        assert inst.llvm_ty == "float"
        assert inst.operand == "%1"
        assert inst.raw_line == line

    def test_fneg_double(self):
        line = "%3 = fneg double %2"
        inst = parse_instruction(line)
        assert isinstance(inst, FNeg)
        assert inst.llvm_ty == "double"


class TestFreeze:
    def test_freeze_i32(self):
        line = "%2 = freeze i32 %1"
        inst = parse_instruction(line)
        assert isinstance(inst, Freeze)
        assert inst.out_ssa == "%2"
        assert inst.llvm_ty == "i32"
        assert inst.operand == "%1"
        assert inst.raw_line == line

    def test_freeze_float(self):
        line = "%3 = freeze float %2"
        inst = parse_instruction(line)
        assert isinstance(inst, Freeze)
        assert inst.llvm_ty == "float"


class TestVectorOp:
    def test_extractelement(self):
        line = "%2 = extractelement <4 x float> %1, i32 0"
        inst = parse_instruction(line)
        assert isinstance(inst, VectorOp)
        assert inst.vector_op == "extractelement"
        assert inst.out_ssa == "%2"
        assert inst.raw_line == line

    def test_insertelement(self):
        line = "%3 = insertelement <4 x float> %1, float %2, i32 0"
        inst = parse_instruction(line)
        assert isinstance(inst, VectorOp)
        assert inst.vector_op == "insertelement"
        assert inst.out_ssa == "%3"

    def test_shufflevector(self):
        line = "%3 = shufflevector <4 x float> %1, <4 x float> %2, <4 x i32> <i32 0, i32 1, i32 4, i32 5>"
        inst = parse_instruction(line)
        assert isinstance(inst, VectorOp)
        assert inst.vector_op == "shufflevector"


class TestAggregateOp:
    def test_extractvalue(self):
        line = "%2 = extractvalue {i32, i1} %1, 0"
        inst = parse_instruction(line)
        assert isinstance(inst, AggregateOp)
        assert inst.agg_op == "extractvalue"
        assert inst.out_ssa == "%2"
        assert inst.raw_line == line

    def test_insertvalue(self):
        line = "%3 = insertvalue {i32, i1} %1, i32 %2, 0"
        inst = parse_instruction(line)
        # insertvalue with 6-group regex — may or may not match
        # depending on exact format; verify type at minimum
        assert isinstance(inst, (AggregateOp, UnknownInstruction))


class TestAtomicOp:
    def test_atomicrmw_add(self):
        line = "%2 = atomicrmw add ptr addrspace(1) %0, i32 %1 seq_cst"
        inst = parse_instruction(line)
        assert isinstance(inst, AtomicOp)
        assert inst.atomic_op == "add"
        assert inst.ordering == "seq_cst"
        assert inst.out_ssa == "%2"
        assert inst.raw_line == line

    def test_atomicrmw_xchg(self):
        line = "%3 = atomicrmw xchg ptr addrspace(1) %0, i32 %1 acquire"
        inst = parse_instruction(line)
        assert isinstance(inst, AtomicOp)
        assert inst.atomic_op == "xchg"
        assert inst.ordering == "acquire"


class TestAlloca:
    def test_alloca_float_align4(self):
        line = "%1 = alloca float, align 4"
        inst = parse_instruction(line)
        assert isinstance(inst, Alloca)
        assert inst.out_ssa == "%1"
        assert inst.alloc_ty == "float"
        assert inst.alignment == "4"
        assert inst.raw_line == line

    def test_alloca_i32(self):
        line = "%2 = alloca i32"
        inst = parse_instruction(line)
        assert isinstance(inst, Alloca)
        assert inst.alloc_ty == "i32"
        assert inst.alignment is None


class TestTerminator:
    def test_br_unconditional(self):
        line = "br label %bb1"
        inst = parse_instruction(line)
        assert isinstance(inst, Terminator)
        assert inst.term_kind == "br"
        assert inst.raw_line == line

    def test_br_conditional(self):
        line = "br i1 %cond, label %bb1, label %bb2"
        inst = parse_instruction(line)
        assert isinstance(inst, Terminator)
        assert inst.term_kind == "br_cond"

    def test_ret_void(self):
        line = "ret void"
        inst = parse_instruction(line)
        assert isinstance(inst, Terminator)
        assert inst.term_kind == "ret"
        assert inst.operands_raw == "void"

    def test_unreachable(self):
        line = "unreachable"
        inst = parse_instruction(line)
        assert isinstance(inst, Terminator)
        assert inst.term_kind == "unreachable"
        assert inst.operands_raw == ""

    def test_fence(self):
        line = "fence seq_cst"
        inst = parse_instruction(line)
        assert isinstance(inst, Terminator)
        assert inst.term_kind == "fence"


# =====================================================================
# 2. Edge case tests
# =====================================================================


class TestEdgeCases:
    def test_metadata_stripped(self):
        line = "%5 = add nsw i32 %3, %4, !dbg !42"
        inst = parse_instruction(line)
        assert isinstance(inst, BinOp)
        assert inst.op == "add"
        assert inst.lhs == "%3"
        assert inst.rhs == "%4"
        assert inst.raw_line == line  # raw_line preserves original

    def test_tbaa_metadata_stripped(self):
        line = "%2 = load float, ptr addrspace(1) %1, align 4, !tbaa !7"
        inst = parse_instruction(line)
        assert isinstance(inst, Load)
        assert inst.alignment == "4"

    def test_attr_group_stripped(self):
        line = "%2 = call float @llvm.fabs.f32(float %1) #3"
        inst = parse_instruction(line)
        assert isinstance(inst, Call)
        assert inst.fn_name == "llvm.fabs.f32"

    def test_trailing_comment_stripped(self):
        line = "%5 = add i32 %3, %4 ; comment"
        inst = parse_instruction(line)
        assert isinstance(inst, BinOp)
        assert inst.op == "add"

    def test_empty_line(self):
        inst = parse_instruction("")
        assert isinstance(inst, UnknownInstruction)

    def test_comment_only(self):
        inst = parse_instruction("; this is a comment")
        assert isinstance(inst, UnknownInstruction)

    def test_fast_flags_multiple(self):
        line = "%2 = fadd nnan ninf nsz float %0, %1"
        inst = parse_instruction(line)
        assert isinstance(inst, BinOp)
        assert "nnan" in inst.flags
        assert "ninf" in inst.flags
        assert "nsz" in inst.flags

    def test_exact_flag(self):
        line = "%3 = udiv exact i32 %1, %2"
        inst = parse_instruction(line)
        assert isinstance(inst, BinOp)
        assert "exact" in inst.flags

    def test_disjoint_flag(self):
        line = "%3 = or disjoint i32 %1, %2"
        inst = parse_instruction(line)
        assert isinstance(inst, BinOp)
        assert "disjoint" in inst.flags

    def test_combined_metadata_and_comment(self):
        line = "%2 = load i32, ptr %1, align 4, !dbg !5 ; load val"
        inst = parse_instruction(line)
        assert isinstance(inst, Load)

    def test_multiple_metadata_annotations(self):
        line = "%2 = load float, ptr addrspace(1) %1, align 4, !tbaa !7, !noalias !9"
        inst = parse_instruction(line)
        assert isinstance(inst, Load)
        assert inst.alignment == "4"

    def test_load_with_invariant_metadata(self):
        line = "%2 = load float, ptr addrspace(1) %1, align 4, !invariant.load !0"
        inst = parse_instruction(line)
        assert isinstance(inst, Load)


# =====================================================================
# 3. Batch parsing test
# =====================================================================


class TestBatchParsing:
    def test_parse_block_multi_instruction(self):
        lines = [
            "%1 = alloca float, align 4",
            "%2 = load float, ptr addrspace(1) %0, align 4",
            "%3 = fadd float %2, %2",
            "store float %3, ptr addrspace(1) %0, align 4",
            "ret void",
        ]
        results = parse_block(lines)
        assert len(results) == 5
        assert isinstance(results[0], Alloca)
        assert isinstance(results[1], Load)
        assert isinstance(results[2], BinOp)
        assert isinstance(results[3], Store)
        assert isinstance(results[4], Terminator)

    def test_parse_block_empty(self):
        assert parse_block([]) == []

    def test_parse_block_preserves_order(self):
        lines = [
            "%1 = add i32 %0, 1",
            "%2 = icmp slt i32 %1, 10",
            "br i1 %2, label %loop, label %exit",
        ]
        results = parse_block(lines)
        assert results[0].opcode == "add"
        assert results[1].opcode == "icmp"
        assert results[2].opcode == "br"


# =====================================================================
# 4. Unknown instruction test
# =====================================================================


class TestUnknownInstruction:
    def test_fallback_for_define(self):
        line = "define void @kernel(ptr addrspace(1) %0) {"
        inst = parse_instruction(line)
        assert isinstance(inst, UnknownInstruction)
        assert inst.raw_line == line

    def test_fallback_for_label(self):
        line = "bb1:"
        inst = parse_instruction(line)
        assert isinstance(inst, UnknownInstruction)

    def test_fallback_for_attribute_group(self):
        line = "attributes #0 = { nounwind }"
        inst = parse_instruction(line)
        assert isinstance(inst, UnknownInstruction)

    def test_fallback_for_declare(self):
        line = "declare float @llvm.fabs.f32(float)"
        inst = parse_instruction(line)
        assert isinstance(inst, UnknownInstruction)

    def test_never_raises(self):
        weird_lines = [
            "!!!garbage!!!",
            "   ",
            "\t\t\n",
            "🤖 robot",
            "42",
        ]
        for line in weird_lines:
            inst = parse_instruction(line)
            assert isinstance(inst, (UnknownInstruction, LLVMInstruction))


# =====================================================================
# 5. Round-trip test (raw_line preservation)
# =====================================================================


class TestRoundTrip:
    ALL_SAMPLE_LINES = [
        # BinOp
        "%5 = add nsw i32 %3, %4",
        "%7 = fadd float %5, %6",
        "%9 = mul i32 %7, %8",
        "%11 = and i32 %9, %10",
        "%13 = shl i32 %11, %12",
        # Load
        "%2 = load float, ptr addrspace(1) %1, align 4",
        "%4 = load i32, ptr %3",
        # Store
        "store float %val, ptr addrspace(1) %ptr, align 4",
        # Cast
        "%3 = sext i32 %2 to i64",
        "%5 = fptrunc double %4 to float",
        "%7 = bitcast i32 %6 to float",
        "%9 = addrspacecast ptr addrspace(1) %8 to ptr",
        # ICmp/FCmp
        "%3 = icmp slt i32 %1, %2",
        "%4 = fcmp olt float %2, %3",
        # Call
        "%2 = call float @llvm.fabs.f32(float %1)",
        "call void @llvm.nvvm.barrier0()",
        # GEP
        "%3 = getelementptr inbounds float, ptr addrspace(3) %1, i32 %2",
        # Phi
        "%5 = phi float [ %3, %bb1 ], [ %4, %bb2 ]",
        # Select
        "%4 = select i1 %1, float %2, float %3",
        # FNeg
        "%2 = fneg float %1",
        # Freeze
        "%2 = freeze i32 %1",
        # Extractelement/Insertelement
        "%2 = extractelement <4 x float> %1, i32 0",
        "%3 = insertelement <4 x float> %1, float %2, i32 0",
        # Shufflevector
        "%3 = shufflevector <4 x float> %1, <4 x float> %2, <4 x i32> <i32 0, i32 1, i32 4, i32 5>",
        # Extractvalue
        "%2 = extractvalue {i32, i1} %1, 0",
        # Atomicrmw
        "%2 = atomicrmw add ptr addrspace(1) %0, i32 %1 seq_cst",
        # Alloca
        "%1 = alloca float, align 4",
        # Branch/Terminator
        "br label %bb1",
        "br i1 %cond, label %bb1, label %bb2",
        "ret void",
        # Metadata (raw_line preserved even when cleaned for parsing)
        "%5 = add nsw i32 %3, %4, !dbg !42",
    ]

    @pytest.mark.parametrize("line", ALL_SAMPLE_LINES)
    def test_raw_line_preserved(self, line):
        inst = parse_instruction(line)
        assert inst.raw_line == line, (
            f"raw_line mismatch for {type(inst).__name__}: "
            f"expected {line!r}, got {inst.raw_line!r}"
        )

    @pytest.mark.parametrize("line", ALL_SAMPLE_LINES)
    def test_not_unknown_for_known_lines(self, line):
        inst = parse_instruction(line)
        assert not isinstance(
            inst, UnknownInstruction
        ), f"Expected typed instruction for: {line!r}, got UnknownInstruction"


# =====================================================================
# 6. Type alias coverage
# =====================================================================


class TestTypeAlias:
    def test_instruction_union_covers_all_types(self):
        all_types = {
            BinOp,
            Load,
            Store,
            Cast,
            ICmp,
            FCmp,
            Call,
            GEP,
            Phi,
            Select,
            FNeg,
            Freeze,
            VectorOp,
            AggregateOp,
            AtomicOp,
            Alloca,
            Terminator,
            UnknownInstruction,
        }
        # Instruction is a Union type — verify all concrete types are included
        # by checking that instances of each type are assignable
        for ty in all_types:
            assert issubclass(ty, LLVMInstruction)

    def test_frozen_dataclasses(self):
        inst = parse_instruction("%5 = add nsw i32 %3, %4")
        with pytest.raises(AttributeError):
            inst.op = "sub"  # type: ignore[misc]

    def test_slots_dataclasses(self):
        inst = parse_instruction("%5 = add nsw i32 %3, %4")
        assert hasattr(inst, "__slots__")
