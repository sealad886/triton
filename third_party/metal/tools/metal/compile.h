#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <stdint.h>

#ifdef __APPLE__
#import <Metal/Metal.h>
#endif

typedef void *TT_StreamTy;
typedef int TT_ResultTy;
typedef void *MTLBufferPtr;

enum {{
  TT_METAL_SUCCESS = 0,
  TT_METAL_ERROR_INVALID_VALUE = 1,
  TT_METAL_ERROR_RUNTIME = 2,
}};

#endif

// tt-linker-backend: {backend_name}

void unload_{kernel_name}(void);
void load_{kernel_name}(void);
// tt-linker: {kernel_name}:{full_signature}:{algo_info}
TT_ResultTy{_placeholder} {kernel_name}(TT_StreamTy stream, {signature});
