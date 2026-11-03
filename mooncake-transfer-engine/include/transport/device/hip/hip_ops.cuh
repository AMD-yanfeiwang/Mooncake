// HIP implementations of the device-side memory ordering primitives used by
// Mooncake PG. Uses __hip_atomic_* builtins rather than inline ISA so the code
// is valid on every CDNA target.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace mooncake {
namespace device {

__device__ __forceinline__ uint64_t mc_ld_acquire_u64(const uint64_t* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void mc_st_release_u64(const uint64_t* ptr,
                                                  uint64_t val) {
    __hip_atomic_store(const_cast<uint64_t*>(ptr), val, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
}

}  // namespace device
}  // namespace mooncake
