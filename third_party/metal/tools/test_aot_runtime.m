/* clang-format off */
/**
 * Metal AOT Runtime Test Harness
 *
 * Loads a precompiled .metallib, creates a compute pipeline, dispatches
 * a vector_add kernel, reads back results, and verifies correctness.
 *
 * Build (macOS):
 *   clang -framework Metal -framework Foundation -framework CoreGraphics \
 *         -o test_aot_runtime test_aot_runtime.m
 *
 * Usage:
 *   ./test_aot_runtime <path-to.metallib> [kernel_name] [num_elements]
 *
 * Default kernel_name:  vector_add_kernel
 * Default num_elements: 1024
 *
 * The kernel is expected to take three device float* buffers (A, B, out)
 * and a uint element-count, computing out[i] = A[i] + B[i].
 *
 * Exit code 0 = PASS, 1 = FAIL.
 */

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void die(const char *msg) {
    fprintf(stderr, "FAIL: %s\n", msg);
    exit(1);
}

static void die_ns(NSString *msg) {
    fprintf(stderr, "FAIL: %s\n", [msg UTF8String]);
    exit(1);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <metallib> [kernel_name] [num_elements]\n", argv[0]);
            return 1;
        }

        const char *metallib_path = argv[1];
        const char *kernel_name = (argc >= 3) ? argv[2] : "vector_add_kernel";
        int num_elements = (argc >= 4) ? atoi(argv[3]) : 1024;
        if (num_elements <= 0) num_elements = 1024;

        printf("AOT Runtime Harness\n");
        printf("  metallib:     %s\n", metallib_path);
        printf("  kernel:       %s\n", kernel_name);
        printf("  num_elements: %d\n", num_elements);

        /* ── 1. Get Metal device ── */
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) die("No Metal device available");
        printf("  device:       %s\n", [[device name] UTF8String]);

        /* ── 2. Load metallib ── */
        NSString *libPath = [NSString stringWithUTF8String:metallib_path];
        NSURL *libURL = [NSURL fileURLWithPath:libPath];
        NSError *libError = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&libError];
        if (library == nil) {
            NSString *msg = [NSString stringWithFormat:@"Failed to load metallib: %@",
                             libError ? [libError localizedDescription] : @"unknown"];
            die_ns(msg);
        }
        printf("  library:      loaded (%lu functions)\n",
               (unsigned long)[[library functionNames] count]);

        /* ── 3. Get kernel function ── */
        NSString *fnName = [NSString stringWithUTF8String:kernel_name];
        id<MTLFunction> function = [library newFunctionWithName:fnName];
        if (function == nil) {
            NSString *msg = [NSString stringWithFormat:@"Kernel '%@' not found. Available: %@",
                             fnName, [library functionNames]];
            die_ns(msg);
        }

        /* ── 4. Create compute pipeline ── */
        NSError *pipeError = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&pipeError];
        if (pipeline == nil) {
            NSString *msg = [NSString stringWithFormat:@"Pipeline creation failed: %@",
                             pipeError ? [pipeError localizedDescription] : @"unknown"];
            die_ns(msg);
        }
        printf("  pipeline:     created (maxThreads=%lu)\n",
               (unsigned long)[pipeline maxTotalThreadsPerThreadgroup]);

        /* ── 5. Create buffers ── */
        size_t buf_size = (size_t)num_elements * sizeof(float);
        id<MTLBuffer> bufA = [device newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [device newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufOut) die("Failed to allocate MTLBuffers");

        /* Fill input arrays: A[i] = i*1.0, B[i] = i*2.0 */
        float *ptrA = (float *)[bufA contents];
        float *ptrB = (float *)[bufB contents];
        for (int i = 0; i < num_elements; i++) {
            ptrA[i] = (float)i;
            ptrB[i] = (float)(i * 2);
        }

        /* ── 6. Encode and dispatch ── */
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) die("Failed to create command queue");

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        if (cmdBuf == nil) die("Failed to create command buffer");

        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        if (encoder == nil) die("Failed to create compute encoder");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufOut offset:0 atIndex:2];

        /* Set n_elements as a scalar argument at buffer index 3 */
        uint32_t n_elem = (uint32_t)num_elements;
        [encoder setBytes:&n_elem length:sizeof(n_elem) atIndex:3];

        NSUInteger threads_per_tg = [pipeline maxTotalThreadsPerThreadgroup];
        if (threads_per_tg > (NSUInteger)num_elements) {
            threads_per_tg = (NSUInteger)num_elements;
        }
        /* Grid: ceil(num_elements / threads_per_tg) threadgroups */
        NSUInteger num_tg = ((NSUInteger)num_elements + threads_per_tg - 1) / threads_per_tg;

        MTLSize tgSize = MTLSizeMake(threads_per_tg, 1, 1);
        MTLSize gridSize = MTLSizeMake(num_tg, 1, 1);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if ([cmdBuf status] == MTLCommandBufferStatusError) {
            NSError *cmdErr = [cmdBuf error];
            NSString *msg = [NSString stringWithFormat:@"Command buffer error: %@",
                             cmdErr ? [cmdErr localizedDescription] : @"unknown"];
            die_ns(msg);
        }

        /* ── 7. Validate results ── */
        float *ptrOut = (float *)[bufOut contents];
        int errors = 0;
        float max_err = 0.0f;
        for (int i = 0; i < num_elements; i++) {
            float expected = ptrA[i] + ptrB[i];  /* i + 2i = 3i */
            float diff = fabsf(ptrOut[i] - expected);
            if (diff > 1e-5f) {
                if (errors < 5) {
                    fprintf(stderr, "  mismatch [%d]: expected=%.6f got=%.6f diff=%.6e\n",
                            i, expected, ptrOut[i], diff);
                }
                errors++;
            }
            if (diff > max_err) max_err = diff;
        }

        printf("  max_error:    %.6e\n", max_err);
        printf("  mismatches:   %d / %d\n", errors, num_elements);

        if (errors == 0) {
            printf("RESULT: PASS\n");
            return 0;
        } else {
            printf("RESULT: FAIL\n");
            return 1;
        }
    }
}

#else /* not __APPLE__ */

#include <stdio.h>

int main(int argc, const char *argv[]) {
    (void)argc; (void)argv;
    fprintf(stderr, "SKIP: Metal AOT runtime harness requires macOS\n");
    return 0;
}

#endif
