#include "NvidiaArtifactLowering.h"
#include "LoadStoreOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <gtest/gtest.h>

namespace mlir::triton::Metal {
namespace {

class NvidiaArtifactLoweringTest : public ::testing::Test {
protected:
  NvidiaArtifactLoweringTest() {
    ctx.loadDialect<LLVM::LLVMDialect>();
  }

  ModuleOp createEmptyModule() {
    OpBuilder builder(&ctx);
    module = ModuleOp::create(builder.getUnknownLoc());
    return module;
  }

  MLIRContext ctx;
  ModuleOp module;
};

TEST_F(NvidiaArtifactLoweringTest, EnsureSharedMemorySymbolCreatesGlobal) {
  auto mod = createEmptyModule();
  ASSERT_FALSE(mod.lookupSymbol("global_smem"));

  ensureSharedMemorySymbol(mod, 3);

  auto sym = mod.lookupSymbol("global_smem");
  ASSERT_TRUE(sym != nullptr);

  auto globalOp = dyn_cast<LLVM::GlobalOp>(sym);
  ASSERT_TRUE(globalOp != nullptr);
  EXPECT_EQ(globalOp.getAddrSpace(), 3u);

  mod->destroy();
}

TEST_F(NvidiaArtifactLoweringTest, EnsureSharedMemorySymbolIdempotent) {
  auto mod = createEmptyModule();

  ensureSharedMemorySymbol(mod, 3);
  ensureSharedMemorySymbol(mod, 3);

  int count = 0;
  for (auto &op : mod.getBody()->getOperations()) {
    if (auto globalOp = dyn_cast<LLVM::GlobalOp>(op)) {
      if (globalOp.getSymName() == "global_smem")
        count++;
    }
  }
  EXPECT_EQ(count, 1);

  mod->destroy();
}

TEST(MangleTypeTest, IntegerTypes) {
  MLIRContext ctx;
  EXPECT_EQ(mangleTypeForSymbol(IntegerType::get(&ctx, 8)), "i8");
  EXPECT_EQ(mangleTypeForSymbol(IntegerType::get(&ctx, 16)), "i16");
  EXPECT_EQ(mangleTypeForSymbol(IntegerType::get(&ctx, 32)), "i32");
  EXPECT_EQ(mangleTypeForSymbol(IntegerType::get(&ctx, 64)), "i64");
}

TEST(MangleTypeTest, FloatTypes) {
  MLIRContext ctx;
  EXPECT_EQ(mangleTypeForSymbol(Float16Type::get(&ctx)), "f16");
  EXPECT_EQ(mangleTypeForSymbol(Float32Type::get(&ctx)), "f32");
  EXPECT_EQ(mangleTypeForSymbol(Float64Type::get(&ctx)), "f64");
}

TEST(MangleTypeTest, PointerType) {
  MLIRContext ctx;
  ctx.loadDialect<LLVM::LLVMDialect>();
  auto ptr0 = LLVM::LLVMPointerType::get(&ctx, 0);
  auto ptr1 = LLVM::LLVMPointerType::get(&ctx, 1);
  EXPECT_EQ(mangleTypeForSymbol(ptr0), "p0");
  EXPECT_EQ(mangleTypeForSymbol(ptr1), "p1");
}

TEST(MangleTypeTest, VoidType) {
  MLIRContext ctx;
  ctx.loadDialect<LLVM::LLVMDialect>();
  EXPECT_EQ(mangleTypeForSymbol(LLVM::LLVMVoidType::get(&ctx)), "void");
}

} // anonymous namespace
} // namespace mlir::triton::Metal
