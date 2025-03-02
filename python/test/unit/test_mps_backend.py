"""
Unit tests for MPS backend Python interface
"""

import unittest
import os
import torch
import triton
import triton.language as tl
import numpy as np

@unittest.skipIf(not torch.backends.mps.is_available(), "MPS not available")
class TestMPSBackend(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("mps")

    def test_device_initialization(self):
        """Test MPS device initialization"""
        self.assertTrue(triton.runtime.driver.MPSDriver.is_active())

        driver = triton.runtime.driver.MPSDriver()
        self.assertEqual(driver.get_current_target()[:3], "mps")
        self.assertEqual(driver.get_active_torch_device(), self.device)

    def test_memory_allocation(self):
        """Test memory allocation and data transfer"""
        size = 1024
        data = torch.ones(size, dtype=torch.float32, device=self.device)

        # Get memory stats before
        stats_before = torch.mps.current_allocated_memory()

        # Perform some allocations
        data2 = torch.ones_like(data)
        data3 = data + data2

        # Get memory stats after
        stats_after = torch.mps.current_allocated_memory()

        # Should have allocated more memory
        self.assertGreater(stats_after, stats_before)

    @unittest.skipIf(not hasattr(torch.mps, 'synchronize'), "MPS synchronize not available")
    def test_synchronization(self):
        """Test device synchronization"""
        driver = triton.runtime.driver.MPSDriver()

        # Perform some computation
        x = torch.ones(1000, device=self.device)
        y = x + 1

        # Should not raise an error
        driver.synchronize()
        self.assertTrue(torch.allclose(y.cpu(), torch.full((1000,), 2.0)))

    def test_kernel_compilation(self):
        """Test simple kernel compilation and execution"""
        @triton.jit
        def add_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)

            output = x + y

            tl.store(z_ptr + offsets, output, mask=mask)

        # Test data
        size = 1024
        x = torch.ones(size, device=self.device)
        y = torch.ones(size, device=self.device)
        z = torch.zeros(size, device=self.device)

        # Launch kernel
        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, z, size, BLOCK_SIZE=128)

        # Verify results
        self.assertTrue(torch.allclose(z.cpu(), torch.full((size,), 2.0)))

    def test_large_tensor(self):
        """Test operations on large tensors"""
        size = 1024 * 1024  # 1M elements
        x = torch.rand(size, device=self.device)
        y = torch.rand(size, device=self.device)

        # Test basic operations
        z = x + y
        w = z * 2.0

        # Verify results
        self.assertEqual(z.shape, (size,))
        self.assertTrue(torch.all(w > z))

    def test_error_handling(self):
        """Test error handling for invalid operations"""
        with self.assertRaises(RuntimeError):
            # Try to allocate too much memory
            x = torch.ones(1024 * 1024 * 1024 * 1024, device=self.device)  # Way too large

    def test_memory_pool(self):
        """Test memory pool functionality"""
        driver = triton.runtime.driver.MPSDriver()
        pool_info = driver.memory_pool()

        self.assertIn('current_allocated', pool_info)
        self.assertIn('max_allocated', pool_info)
        self.assertIn('reserved', pool_info)

        # Values should be non-negative
        self.assertGreaterEqual(pool_info['current_allocated'], 0)
        self.assertGreaterEqual(pool_info['max_allocated'], 0)
        self.assertGreaterEqual(pool_info['reserved'], 0)

    @unittest.skipIf(not hasattr(triton.runtime.driver.MPSDriver, 'get_device_properties'),
                    "Device properties not available")
    def test_device_properties(self):
        """Test device property queries"""
        driver = triton.runtime.driver.MPSDriver()
        props = driver.get_device_properties()

        self.assertIn('name', props)
        self.assertIn('capability', props)
        self.assertIn('total_memory', props)
        self.assertIn('max_threads_per_block', props)
        self.assertIn('max_shared_memory_per_block', props)
        self.assertIn('unified_memory', props)

        # Basic sanity checks
        self.assertTrue(isinstance(props['name'], str))
        self.assertTrue(len(props['name']) > 0)
        self.assertTrue(isinstance(props['capability'], tuple))
        self.assertEqual(len(props['capability']), 2)

    def test_kernel_config(self):
        """Test kernel configuration options"""
        driver = triton.runtime.driver.MPSDriver()

        # Set custom configuration
        driver.set_kernel_config(
            num_warps=8,
            num_stages=2,
            max_threads=512,
            shared_memory=16384
        )

        # Verify configuration was set
        self.assertEqual(driver.config.num_warps, 8)
        self.assertEqual(driver.config.num_stages, 2)
        self.assertEqual(driver.config.max_threads_per_threadgroup, 512)
        self.assertEqual(driver.config.shared_memory_size, 16384)

if __name__ == '__main__':
    unittest.main()
