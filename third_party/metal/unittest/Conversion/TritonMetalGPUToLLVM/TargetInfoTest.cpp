#include "TargetInfo.h"
#include <gtest/gtest.h>

namespace mlir::triton::Metal {
namespace {

TEST(MetalTargetInfoTest, SharedAddressSpaceIs3) {
  TargetInfo info("apple9");
  EXPECT_EQ(info.getSharedAddressSpace(), 3);
}

TEST(MetalTargetInfoTest, NoVectorizedAtomics) {
  TargetInfo info("apple9");
  EXPECT_FALSE(info.supportVectorizedAtomics());
}

TEST(MetalTargetInfoTest, NoMaximumMinimum) {
  TargetInfo info("apple9");
  EXPECT_FALSE(info.supportMaximumMinimum());
}

TEST(MetalTargetInfoTest, ArchStored) {
  TargetInfo info("apple9");
  EXPECT_EQ(info.getArch(), "apple9");

  TargetInfo info7("apple7");
  EXPECT_EQ(info7.getArch(), "apple7");
}

TEST(MetalTargetInfoTest, MulhiFuncName) {
  TargetInfo info("apple9");
  mlir::MLIRContext ctx;
  auto i32Ty = mlir::IntegerType::get(&ctx, 32);
  auto name = info.getMulhiFuncName(i32Ty);
  EXPECT_FALSE(name.empty());
}

} // anonymous namespace
} // namespace mlir::triton::Metal
