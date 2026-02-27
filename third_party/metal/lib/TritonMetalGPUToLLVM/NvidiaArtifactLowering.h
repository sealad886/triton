#ifndef TRITON_METAL_NVIDIAARTIFACTLOWERING_H
#define TRITON_METAL_NVIDIAARTIFACTLOWERING_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::triton::Metal {

void lowerNvidiaArtifactsToMetal(ModuleOp mod);
void ensureSharedMemorySymbol(ModuleOp mod, unsigned addrSpace);

} // namespace mlir::triton::Metal

#endif // TRITON_METAL_NVIDIAARTIFACTLOWERING_H
