cmake_minimum_required(VERSION 3.22)
# find_package before project(): the SDK selects its toolchain before languages are enabled.
# Configure with -DPVA_BUILD_MODE=EMU|L4T -DPVA_DEFAULT_GEN=GEN3 -DPVA_ONLY_BUILD_GENS=GEN3.
find_package(pva-sdk 2.10 REQUIRED)
project(knn1_bucket_s16)

set(CMAKE_CXX_STANDARD 17)
# Position-independent code so the operator can go into the shared C library below.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# VPU program (Part B of the method).
# K1_DIAG_TSC=ON: diagnostic build with per-phase VPU cycle counters (never ship it).
# K1_BUCKET: ref points per bucket (128, 256 or 512); host and VPU are built with the same value.
option(K1_DIAG_TSC "Build the VPU program with per-phase cycle counters" OFF)
set(K1_BUCKET 128 CACHE STRING "Ref points per bucket (128, 256 or 512)")
set(K1_DEVICE_FLAGS -DK1_BUCKET=${K1_BUCKET})
if(K1_DIAG_TSC)
    list(APPEND K1_DEVICE_FLAGS -DK1_DIAG_TSC)
endif()
# Extra VPU compiler flags for experiments (e.g. -DK1_SCAN_VARIANT=1); empty by default.
set(K1_EXTRA_DEVICE_FLAGS "" CACHE STRING "Extra VPU compiler flags (experiments only)")
if(K1_EXTRA_DEVICE_FLAGS)
    list(APPEND K1_DEVICE_FLAGS ${K1_EXTRA_DEVICE_FLAGS})
endif()
pva_device(knn1_bucket_s16_dev vpu/knn1_bucket_s16_dev.c CFLAGS_DEVICE ${K1_DEVICE_FLAGS})

# Host library: API + CPU bucketing (Part A) + embedded VPU program.
add_library_pva(knn1_bucket_s16
    HOST
        src/knn1_bucket_s16.cpp
        src/bucket_prep.cpp
    DEVICE
        knn1_bucket_s16_dev)
target_include_directories(knn1_bucket_s16 PUBLIC include)
find_package(Threads REQUIRED)
target_link_libraries(knn1_bucket_s16 PRIVATE Threads::Threads)
target_compile_definitions(knn1_bucket_s16 PUBLIC K1_BUCKET=${K1_BUCKET})

# Independent brute-force CPU reference (tests and sample only).
add_library(knn1_bucket_s16_ref STATIC ref/knn1_ref.cpp)
target_include_directories(knn1_bucket_s16_ref PUBLIC ref)

add_executable(knn1_bucket_s16_test tests/knn1_bucket_s16_test.cpp)
target_link_libraries(knn1_bucket_s16_test PRIVATE knn1_bucket_s16 knn1_bucket_s16_ref cupva_host)

add_executable(knn1_bucket_s16_perf perf/knn1_bucket_s16_perf.cpp)
target_link_libraries(knn1_bucket_s16_perf PRIVATE knn1_bucket_s16 knn1_bucket_s16_ref cupva_host cupva_host_utils)

add_executable(knn1_bucket_s16_sample sample/knn1_bucket_s16_sample.cpp)
target_link_libraries(knn1_bucket_s16_sample PRIVATE knn1_bucket_s16 knn1_bucket_s16_ref cupva_host)

# Where the PVA time goes (meaningful with K1_DIAG_TSC=ON).
add_executable(knn1_bucket_s16_diag perf/knn1_bucket_s16_diag.cpp)
target_link_libraries(knn1_bucket_s16_diag PRIVATE knn1_bucket_s16 cupva_host cupva_host_utils)

# Cloud distance between two KITTI velodyne scans (.bin files).
add_executable(knn1_bucket_s16_kitti sample/knn1_bucket_s16_kitti.cpp)
target_link_libraries(knn1_bucket_s16_kitti PRIVATE knn1_bucket_s16 cupva_host)

# Shared library with a plain C interface (no cuPVA headers needed by its users).
add_library(knn1_distance SHARED capi/knn1_distance.cpp)
target_include_directories(knn1_distance PUBLIC capi)
target_link_libraries(knn1_distance PRIVATE knn1_bucket_s16 cupva_host)
set_target_properties(knn1_distance PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_link_options(knn1_distance PRIVATE -Wl,--exclude-libs,ALL)

install(TARGETS knn1_bucket_s16_test knn1_bucket_s16_perf knn1_bucket_s16_sample knn1_bucket_s16_kitti
        RUNTIME DESTINATION bin)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../testdata/knn1_bucket DESTINATION testdata)
