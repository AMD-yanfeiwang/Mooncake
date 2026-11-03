// CUDA runtime names used by Mooncake PG, mapped onto HIP. Transfer Engine
// sources are hipified at build time and never see these names.
#pragma once

#include <hip/hip_runtime.h>

// Types
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaGraph_t hipGraph_t
#define cudaHostFn_t hipHostFn_t
#define cudaFuncAttributes hipFuncAttributes
#define cudaStreamCaptureStatus hipStreamCaptureStatus

// Enumerators and flags
#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaHostAllocPortable hipHostMallocPortable
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice

// Functions
#define cudaGetErrorString hipGetErrorString
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaHostAlloc hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamGetCaptureInfo hipStreamGetCaptureInfo
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaFuncGetAttributes hipFuncGetAttributes

// HIP has no typed overload of this function.
template <typename T>
static inline hipError_t cudaHostGetDevicePointer(T** devicePtr, void* hostPtr,
                                                  unsigned int flags) {
    return hipHostGetDevicePointer(reinterpret_cast<void**>(devicePtr),
                                   hostPtr, flags);
}
