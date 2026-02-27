#ifndef METALGPU_ATTRDEFS
#define METALGPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "MetalGPUDialect.td"

class MetalGPU_Attr<string name, list<Trait> traits = [],
                    string baseCppClass = "::mlir::Attribute">
  : AttrDef<MetalGPU_Dialect, name, traits, baseCppClass> {
}

#endif
