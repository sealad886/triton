#ifndef TRITONMETALGPU_TARGETINFO_H
#define TRITONMETALGPU_TARGETINFO_H

#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir {
namespace triton {
namespace Metal {

class TargetInfo : public TargetInfoBase {
public:
  explicit TargetInfo(StringRef arch) : arch(arch.str()) {}

  bool supportMaximumMinimum() const override { return false; }

  Value getClusterCTAId(RewriterBase &rewriter, Location loc) const override;

  Value ballot(RewriterBase &rewriter, Location loc, Type type,
               Value cmp) const override;

  void barrier(Location loc, RewriterBase &rewriter,
               triton::gpu::AddrSpace targets) const override;

  void clusterBarrier(Location loc, RewriterBase &rewriter) const override;

  void warpSync(Location loc, RewriterBase &rewriter) const override;

  void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Value val,
                    Value pred) const override;

  Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Type elemTy, Value pred,
                    Operation *localLoadOp = nullptr) const override;

  Value shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;

  Value shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                  int i) const override;

  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;

  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   Value i) const override;

  Value permute(RewriterBase &rewriter, Location loc, Value a, Value b,
                Value selector) const override;

  Value programId(RewriterBase &rewriter, Location loc, ModuleOp moduleOp,
                  ProgramIDDim axis) const override;

  bool warpReduce(RewriterBase &rewriter, Location loc,
                  SmallVector<Value> &acc, triton::ReduceOp op,
                  unsigned reduceLaneIdMask) const override;

  std::string getMulhiFuncName(Type resultElementTy) const override;

  void printf(RewriterBase &rewriter, Value formatStrStart,
              int formatStrByteCount, ValueRange args,
              ArrayRef<bool> isSigned = {}) const override;

  void printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
              ArrayRef<bool> isSigned = {}) const override;

  void assertFail(RewriterBase &rewriter, Location loc, StringRef message,
                  StringRef file, StringRef func, int line) const override;

  int getSharedAddressSpace() const override;

  int getAddressSpace(Attribute addressSpace) const override;

  bool supportVectorizedAtomics() const override;

  StringRef getArch() const { return arch; }

private:
  std::string arch;
};

} // namespace Metal
} // namespace triton
} // namespace mlir

#endif // TRITONMETALGPU_TARGETINFO_H
