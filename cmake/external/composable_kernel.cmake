# Get all propreties that cmake supports
if(NOT CMAKE_PROPERTY_LIST)
    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)

    # Convert command output into a CMake list
    string(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    string(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    list(REMOVE_DUPLICATES CMAKE_PROPERTY_LIST)
endif()

function(print_properties)
    message("CMAKE_PROPERTY_LIST = ${CMAKE_PROPERTY_LIST}")
endfunction()

function(print_target_properties target)
    if(NOT TARGET ${target})
      message(STATUS "There is no target named '${target}'")
      return()
    endif()

    foreach(property ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" property ${property})

        # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
        if(property STREQUAL "LOCATION" OR property MATCHES "^LOCATION_" OR property MATCHES "_LOCATION$")
            continue()
        endif()

        get_property(was_set TARGET ${target} PROPERTY ${property} SET)
        if(was_set)
            get_target_property(value ${target} ${property})
            message("${target} ${property} = ${value}")
        endif()
    endforeach()
endfunction()

function(remove_hip_device_from_target target)
    # Get the current INTERFACE_LINK_LIBRARIES for the specified target
    get_target_property(current_libs ${target} INTERFACE_LINK_LIBRARIES)

    # Check if hip::device is in the list of libraries
    if(current_libs)
        # Create a list of libraries, removing hip::device if it exists
        string(REPLACE "hip::device" "" modified_libs "${current_libs}")
        # Remove any extra spaces
        string(STRIP "${modified_libs}" modified_libs)

        # Set the modified INTERFACE_LINK_LIBRARIES back to the target
        set_target_properties(${target} PROPERTIES INTERFACE_LINK_LIBRARIES "${modified_libs}")
    endif()
endfunction()

set(GPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})
# ck tile only supports MI200+ GPUs at the moment
string(REPLACE "," ";" GPU_TARGETS "${GPU_TARGETS}")
set(original_archs ${GPU_TARGETS})
list(FILTER GPU_TARGETS INCLUDE REGEX "(gfx942|gfx90a)")
if (NOT original_archs EQUAL GPU_TARGETS)
  message(WARNING "ck tile only supports archs: ${GPU_TARGETS} among the originally specified ${original_archs}")
endif()

add_definitions(-DCK_ENABLE_INT8 -DCK_ENABLE_FP16 -DCK_ENABLE_FP32 -DCK_ENABLE_FP64 -DCK_ENABLE_BF16)
if (GPU_TARGETS MATCHES "gfx94")
    add_definitions(-DCK_ENABLE_FP8 -DCK_ENABLE_BF8)
endif()
add_library(onnxruntime_composable_kernel_includes INTERFACE)
find_package(composable_kernel COMPONENTS device_other_operations device_gemm_operations device_conv_operations  device_reduction_operations)
remove_hip_device_from_target(composable_kernel::device_other_operations)
remove_hip_device_from_target(composable_kernel::device_reduction_operations)
remove_hip_device_from_target(composable_kernel::device_conv_operations)
remove_hip_device_from_target(composable_kernel::device_gemm_operations)
if(GPU_TARGETS MATCHES "gfx9")
    find_package(composable_kernel COMPONENTS device_contraction_operations)
    remove_hip_device_from_target(composable_kernel::device_contraction_operations)
    target_link_libraries(onnxruntime_composable_kernel_includes INTERFACE composable_kernel::device_contraction_operations)
endif()
target_link_libraries(onnxruntime_composable_kernel_includes INTERFACE composable_kernel::device_other_operations
    composable_kernel::device_reduction_operations
    composable_kernel::device_conv_operations
    composable_kernel::device_gemm_operations)
set_target_properties(onnxruntime_composable_kernel_includes PROPERTIES HIP_ARCHITECTURES "${GPU_TARGETS}")

if (onnxruntime_USE_COMPOSABLE_KERNEL_CK_TILE)
 include(FetchContent)
  FetchContent_Declare(composable_kernel
    URL ${DEP_URL_composable_kernel}
    URL_HASH SHA1=${DEP_SHA1_composable_kernel}
  )

  FetchContent_GetProperties(composable_kernel)
  if(NOT composable_kernel_POPULATED)
    FetchContent_Populate(composable_kernel)
    file(REMOVE_RECURSE ${composable_kernel_SOURCE_DIR}/include/ck)

    execute_process(
      COMMAND ${Python3_EXECUTABLE} ${composable_kernel_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
      --list_blobs ${composable_kernel_BINARY_DIR}/blob_list.txt
      COMMAND_ERROR_IS_FATAL ANY
    )
    file(STRINGS ${composable_kernel_BINARY_DIR}/blob_list.txt generated_fmha_srcs)
    add_custom_command(
      OUTPUT ${generated_fmha_srcs}
      COMMAND ${Python3_EXECUTABLE} ${composable_kernel_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py --output_dir ${composable_kernel_BINARY_DIR}
      DEPENDS ${composable_kernel_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py ${composable_kernel_BINARY_DIR}/blob_list.txt
    )
    set_source_files_properties(${generated_fmha_srcs} PROPERTIES LANGUAGE HIP GENERATED TRUE)
    add_custom_target(gen_fmha_srcs DEPENDS ${generated_fmha_srcs})  # dummy target for dependencies
    # code generation complete
    add_library(onnxruntime_composable_kernel_fmha STATIC EXCLUDE_FROM_ALL ${generated_fmha_srcs})
    target_link_libraries(onnxruntime_composable_kernel_fmha PUBLIC onnxruntime_composable_kernel_includes)
    target_include_directories(onnxruntime_composable_kernel_fmha PUBLIC ${composable_kernel_SOURCE_DIR}/example/ck_tile/01_fmha
      ${composable_kernel_SOURCE_DIR}/include)
    add_dependencies(onnxruntime_composable_kernel_fmha gen_fmha_srcs)
    set_target_properties(onnxruntime_composable_kernel_fmha PROPERTIES HIP_ARCHITECTURES "${GPU_TARGETS}")
  endif()
endif()
