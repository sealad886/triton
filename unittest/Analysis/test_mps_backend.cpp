#include "metal/metal_buffer.h"
#include "metal/metal_device.h"
#include "metal/metal_runtime.h"
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

using namespace triton::backend::mps;

class MPSBackendTest : public ::testing::Test {
protected:
  void SetUp() override {
    runtime_ = new MetalRuntime();
    ASSERT_TRUE(runtime_->initialize()) << "Failed to initialize Metal runtime";
    device_ = runtime_->getDevice();
    ASSERT_NE(device_, nullptr) << "Failed to get Metal device";
  }

  void TearDown() override { delete runtime_; }

  // Helper function to create and fill a buffer
  MetalBuffer *createBuffer(size_t size, float fill_value = 1.0f) {
    MetalBuffer *buffer = new MetalBuffer();
    std::vector<float> host_data(size, fill_value);
    EXPECT_TRUE(device_->createBuffer(size * sizeof(float), *buffer));
    EXPECT_TRUE(buffer->copyFromHost(host_data.data(), size * sizeof(float)));
    return buffer;
  }

  // Helper function to verify buffer contents
  bool verifyBuffer(MetalBuffer *buffer, float expected_value, size_t size) {
    std::vector<float> host_data(size);
    if (!buffer->copyToHost(host_data.data(), size * sizeof(float))) {
      return false;
    }
    for (size_t i = 0; i < size; ++i) {
      if (std::abs(host_data[i] - expected_value) > 1e-6) {
        return false;
      }
    }
    return true;
  }

  MetalRuntime *runtime_;
  MetalDevice *device_;
};

TEST_F(MPSBackendTest, DeviceCapabilities) {
  const auto &caps = device_->getCapabilities();
  EXPECT_GT(caps.max_threads_per_threadgroup, 0);
  EXPECT_GT(caps.shared_memory_size, 0);
  EXPECT_GT(caps.max_buffer_size, 0);
}

TEST_F(MPSBackendTest, BufferOperations) {
  const size_t size = 1024;
  const float value = 42.0f;

  // Create and fill buffer
  auto buffer = createBuffer(size, value);
  ASSERT_NE(buffer, nullptr);

  // Verify contents
  EXPECT_TRUE(verifyBuffer(buffer, value, size));

  // Test mapping
  float *mapped = static_cast<float *>(buffer->map());
  ASSERT_NE(mapped, nullptr);

  // Modify through mapping
  const float new_value = 24.0f;
  for (size_t i = 0; i < size; ++i) {
    mapped[i] = new_value;
  }
  buffer->unmap();

  // Verify new contents
  EXPECT_TRUE(verifyBuffer(buffer, new_value, size));

  delete buffer;
}

TEST_F(MPSBackendTest, MatrixMultiplication) {
  const int M = 32;
  const int N = 32;
  const int K = 32;

  // Create input matrices
  auto A = createBuffer(M * K, 1.0f);
  auto B = createBuffer(K * N, 1.0f);
  auto C = createBuffer(M * N, 0.0f);

  ASSERT_NE(A, nullptr);
  ASSERT_NE(B, nullptr);
  ASSERT_NE(C, nullptr);

  // Perform matrix multiplication
  EXPECT_TRUE(runtime_->matmul(A, B, C, M, N, K));

  // Verify result (each element should be K)
  EXPECT_TRUE(verifyBuffer(C, static_cast<float>(K), M * N));

  delete A;
  delete B;
  delete C;
}

TEST_F(MPSBackendTest, KernelCompilation) {
  const char *kernel_source = R"(
    kernel void test_kernel(
      device float* input [[buffer(0)]],
      device float* output [[buffer(1)]],
      uint index [[thread_position_in_grid]]
    ) {
      output[index] = input[index] * 2.0f;
    }
  )";

  // Compile kernel
  MetalShader shader;
  EXPECT_TRUE(runtime_->compileKernel(kernel_source, "test_kernel", shader));

  // Create buffers
  const size_t size = 128;
  auto input = createBuffer(size, 1.0f);
  auto output = createBuffer(size, 0.0f);

  // Launch kernel
  std::vector<MetalBuffer *> args = {input, output};
  dim3 grid(size);
  dim3 block(1);
  EXPECT_TRUE(runtime_->launchKernel(shader, args, grid, block));

  // Verify result
  EXPECT_TRUE(verifyBuffer(output, 2.0f, size));

  delete input;
  delete output;
}

TEST_F(MPSBackendTest, DeviceSynchronization) {
  // Create buffers for concurrent operations
  const size_t size = 1024;
  std::vector<MetalBuffer *> buffers;

  // Launch multiple operations
  for (int i = 0; i < 5; ++i) {
    auto buffer = createBuffer(size, static_cast<float>(i));
    buffers.push_back(buffer);
  }

  // Synchronize device
  EXPECT_TRUE(runtime_->deviceSynchronize());

  // Verify all operations completed
  for (size_t i = 0; i < buffers.size(); ++i) {
    EXPECT_TRUE(verifyBuffer(buffers[i], static_cast<float>(i), size));
    delete buffers[i];
  }
}
