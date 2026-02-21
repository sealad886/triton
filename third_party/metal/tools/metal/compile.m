/* clang-format off */
#include "compile.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef __APPLE__
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

// globals
#define METALLIB_NAME {kernel_name}_metallib
static id<MTLDevice> {kernel_name}_dev = nil;
static id<MTLCommandQueue> {kernel_name}_queue = nil;
static id<MTLLibrary> {kernel_name}_mod = nil;
static id<MTLComputePipelineState> {kernel_name}_func = nil;
unsigned char METALLIB_NAME[{bin_size}] = {{ {bin_data} }};

static inline TT_ResultTy tt_metal_error(const char *msg) {{
  const char *prefix = "Triton Error [Metal]: ";
  char err[1024] = {{0}};
  strcat(err, prefix);
  strcat(err, msg);
  fprintf(stderr, "%s\n", err);
  return TT_METAL_ERROR_RUNTIME;
}}

void unload_{kernel_name}(void) {{
  {kernel_name}_func = nil;
  {kernel_name}_mod = nil;
  {kernel_name}_queue = nil;
  {kernel_name}_dev = nil;
}}

void load_{kernel_name}(void) {{
  {kernel_name}_dev = MTLCreateSystemDefaultDevice();
  if ({kernel_name}_dev == nil) {{
    return;
  }}
  {kernel_name}_queue = [{kernel_name}_dev newCommandQueue];
  if ({kernel_name}_queue == nil) {{
    unload_{kernel_name}();
    return;
  }}

  dispatch_data_t lib_data = dispatch_data_create(
      METALLIB_NAME,
      sizeof(METALLIB_NAME),
      dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
      DISPATCH_DATA_DESTRUCTOR_DEFAULT);

  NSError *library_error = nil;
  {kernel_name}_mod = [{kernel_name}_dev newLibraryWithData:lib_data error:&library_error];
  if ({kernel_name}_mod == nil || library_error != nil) {{
    unload_{kernel_name}();
    return;
  }}

  id<MTLFunction> kernel_fn = [{kernel_name}_mod newFunctionWithName:@"{triton_kernel_name}"];
  if (kernel_fn == nil) {{
    unload_{kernel_name}();
    return;
  }}

  NSError *pipeline_error = nil;
  {kernel_name}_func = [{kernel_name}_dev newComputePipelineStateWithFunction:kernel_fn error:&pipeline_error];
  if ({kernel_name}_func == nil || pipeline_error != nil) {{
    unload_{kernel_name}();
    return;
  }}
}}

/*
{kernel_docstring}
*/
TT_ResultTy {kernel_name}(TT_StreamTy stream, {signature}) {{
  (void)stream;

  if ({kernel_name}_func == nil || {kernel_name}_queue == nil) {{
    load_{kernel_name}();
  }}
  if ({kernel_name}_func == nil || {kernel_name}_queue == nil) {{
    return tt_metal_error("failed to initialize Metal pipeline");
  }}

  unsigned int gX = {gridX};
  unsigned int gY = {gridY};
  unsigned int gZ = {gridZ};
  if (gX == 0 || gY == 0 || gZ == 0) {{
    return TT_METAL_ERROR_INVALID_VALUE;
  }}

  id<MTLCommandBuffer> cmd_buf = [{kernel_name}_queue commandBuffer];
  if (cmd_buf == nil) {{
    return tt_metal_error("failed to allocate command buffer");
  }}

  id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
  if (encoder == nil) {{
    return tt_metal_error("failed to allocate command encoder");
  }}
  [encoder setComputePipelineState:{kernel_name}_func];

{metal_arg_bindings}

  NSUInteger threads_per_threadgroup = {metal_threads_per_threadgroup};
  if (threads_per_threadgroup == 0) {{
    threads_per_threadgroup = 1;
  }}
  if (threads_per_threadgroup > [{kernel_name}_func maxTotalThreadsPerThreadgroup]) {{
    threads_per_threadgroup = [{kernel_name}_func maxTotalThreadsPerThreadgroup];
  }}

  MTLSize tg_size = MTLSizeMake(threads_per_threadgroup, 1, 1);
  MTLSize tg_count = MTLSizeMake(gX, gY, gZ);
  [encoder dispatchThreadgroups:tg_count threadsPerThreadgroup:tg_size];
  [encoder endEncoding];

  [cmd_buf commit];
  [cmd_buf waitUntilCompleted];
  if ([cmd_buf status] == MTLCommandBufferStatusError) {{
    NSError *error = [cmd_buf error];
    if (error != nil) {{
      return tt_metal_error([[error localizedDescription] UTF8String]);
    }}
    return tt_metal_error("command buffer failed");
  }}
  return TT_METAL_SUCCESS;
}}

#else

void unload_{kernel_name}(void) {{
}}

void load_{kernel_name}(void) {{
}}

TT_ResultTy {kernel_name}(TT_StreamTy stream, {signature}) {{
  (void)stream;
  return TT_METAL_ERROR_RUNTIME;
}}

#endif
