/**
 * Metal backend native utilities for Triton.
 *
 * Provides Python-accessible functions for querying Metal device properties
 * and loading compiled Metal libraries. Built as a Python C extension module
 * using the Metal framework on macOS.
 */

#include <stdbool.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef __APPLE__
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
    int device_id;
    if (!PyArg_ParseTuple(args, "i", &device_id))
        return NULL;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        PyErr_SetString(PyExc_RuntimeError, "No Metal device available");
        return NULL;
    }

    NSUInteger maxThreads = device.maxThreadsPerThreadgroup.width
                          * device.maxThreadsPerThreadgroup.height
                          * device.maxThreadsPerThreadgroup.depth;

    return Py_BuildValue(
        "{s:s, s:L, s:L, s:L, s:i}",
        "name", [device.name UTF8String],
        "max_buffer_length", (long long)device.maxBufferLength,
        "max_threads_per_threadgroup", (long long)maxThreads,
        "max_threadgroup_memory_length", (long long)device.maxThreadgroupMemoryLength,
        "supports_family_apple7", (int)[device supportsFamily:MTLGPUFamilyApple7]
    );
}

static PyObject *isMetalAvailable(PyObject *self, PyObject *args) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device != nil) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *getDeviceName(PyObject *self, PyObject *args) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return PyUnicode_FromString("unknown");
    }
    return PyUnicode_FromString([device.name UTF8String]);
}

static PyObject *getMaxThreadgroupSize(PyObject *self, PyObject *args) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return PyLong_FromLong(0);
    }
    MTLSize size = device.maxThreadsPerThreadgroup;
    return Py_BuildValue("(LLL)",
        (long long)size.width,
        (long long)size.height,
        (long long)size.depth);
}

static PyObject *loadMetallib(PyObject *self, PyObject *args) {
    Py_buffer buffer;
    if (!PyArg_ParseTuple(args, "y*", &buffer))
        return NULL;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        PyBuffer_Release(&buffer);
        PyErr_SetString(PyExc_RuntimeError, "No Metal device available");
        return NULL;
    }

    NSData *data = [NSData dataWithBytes:buffer.buf length:buffer.len];
    PyBuffer_Release(&buffer);

    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithData:dispatch_data_create(
        data.bytes, data.length, nil, nil) error:&error];

    if (error != nil || library == nil) {
        const char *errMsg = error ? [[error localizedDescription] UTF8String] : "Unknown error";
        PyErr_Format(PyExc_RuntimeError, "Failed to load metallib: %s", errMsg);
        return NULL;
    }

    NSArray<NSString *> *functionNames = [library functionNames];
    PyObject *nameList = PyList_New([functionNames count]);
    for (NSUInteger i = 0; i < [functionNames count]; i++) {
        PyList_SetItem(nameList, i, PyUnicode_FromString([functionNames[i] UTF8String]));
    }

    return Py_BuildValue("{s:O, s:i}",
        "function_names", nameList,
        "num_functions", (int)[functionNames count]);
}

#else /* !__APPLE__ */

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
    return Py_BuildValue(
        "{s:s, s:i, s:i, s:i, s:i}",
        "name", "unavailable",
        "max_buffer_length", 0,
        "max_threads_per_threadgroup", 0,
        "max_threadgroup_memory_length", 0,
        "supports_family_apple7", 0
    );
}

static PyObject *isMetalAvailable(PyObject *self, PyObject *args) {
    Py_RETURN_FALSE;
}

static PyObject *getDeviceName(PyObject *self, PyObject *args) {
    return PyUnicode_FromString("unavailable");
}

static PyObject *getMaxThreadgroupSize(PyObject *self, PyObject *args) {
    return Py_BuildValue("(iii)", 0, 0, 0);
}

static PyObject *loadMetallib(PyObject *self, PyObject *args) {
    PyErr_SetString(PyExc_RuntimeError, "Metal is not available on this platform");
    return NULL;
}

#endif /* __APPLE__ */

static PyMethodDef ModuleMethods[] = {
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get Metal device properties for the given device ID"},
    {"is_metal_available", isMetalAvailable, METH_NOARGS,
     "Check if Metal is available on this system"},
    {"get_device_name", getDeviceName, METH_NOARGS,
     "Get the name of the default Metal device"},
    {"get_max_threadgroup_size", getMaxThreadgroupSize, METH_NOARGS,
     "Get the maximum threadgroup size as (width, height, depth)"},
    {"load_metallib", loadMetallib, METH_VARARGS,
     "Load a .metallib binary and return function names"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "metal_utils",
    "Metal GPU utilities for Triton",
    -1,
    ModuleMethods
};

PyMODINIT_FUNC PyInit_metal_utils(void) {
    PyObject *m = PyModule_Create(&ModuleDef);
    if (m == NULL)
        return NULL;
    return m;
}
