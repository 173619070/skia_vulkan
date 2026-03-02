cmake_minimum_required(VERSION 3.10)

# Set policy before project() to affect compiler flags initialization
cmake_policy(SET CMP0091 NEW)

project(SkiaVulkanExample)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure MSVC Runtime Library to /MD (matches wxWidgets defaults)
# We force /MD even for Debug builds to avoid mismatches with Release-built Skia and wxWidgets
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

# Fallback in case CMP0091 isn't fully working for older CMake or specific generators
string(REPLACE "/MDd" "/MD" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
string(REPLACE "/MTd" "/MD" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
string(REPLACE "/MT" "/MD" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

# Force iterator debug level to 0 to match Skia Release build
add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)

# Skia root directory
set(SKIA_ROOT "E:/skia")
set(SKIA_OUT_DIR "${SKIA_ROOT}/out/Static")

# Vulkan SDK directory
set(VULKAN_SDK "C:/VulkanSDK/1.4.304.1")

# Add Skia includes
include_directories(${SKIA_ROOT})
include_directories("${SKIA_ROOT}/third_party/externals/vulkanmemoryallocator/include")

# Add Vulkan includes
include_directories("${VULKAN_SDK}/Include")

# Definitions for Skia Vulkan backend and Windows macro collision avoidance
add_definitions(-DSK_VULKAN -DSK_SUPPORT_GPU=1 -DSK_GL=1 -DSK_GANESH -DVK_USE_PLATFORM_WIN32_KHR -DNOMINMAX -DWIN32_LEAN_AND_MEAN)

# Target executable
add_executable(VulkanBasic 
    VulkanBasic.cpp
    "${SKIA_ROOT}/tools/gpu/vk/VkTestUtils.cpp"
    "${SKIA_ROOT}/tools/gpu/vk/VkTestHelper.cpp"
    "${SKIA_ROOT}/tools/gpu/vk/VkTestMemoryAllocator.cpp"
    "${SKIA_ROOT}/src/gpu/vk/VulkanExtensions.cpp"
    "${SKIA_ROOT}/src/gpu/vk/VulkanPreferredFeatures.cpp"
    "${SKIA_ROOT}/src/gpu/vk/VulkanInterface.cpp"
    "${SKIA_ROOT}/src/gpu/vk/vulkanmemoryallocator/VulkanAMDMemoryAllocator.cpp"
    "${SKIA_ROOT}/third_party/externals/vulkanmemoryallocator/src/VmaUsage.cpp"
)

# Define Skia static library dependencies
find_library(SKIA_LIB NAMES skia PATHS ${SKIA_OUT_DIR} NO_DEFAULT_PATH)
find_library(VULKAN_LIB NAMES vulkan-1 PATHS "${VULKAN_SDK}/Lib" NO_DEFAULT_PATH)

# Additional Skia libraries that might contain Ganesh/Vulkan components
find_library(SKIA_GANESH_GL_LIB NAMES skia_gl PATHS ${SKIA_OUT_DIR} NO_DEFAULT_PATH)
find_library(SKIA_GANESH_VK_LIB NAMES skia_vulkan PATHS ${SKIA_OUT_DIR} NO_DEFAULT_PATH)
# Some Skia builds merge everything into skia.lib, others split it.
# If GrDirectContexts::MakeVulkan is missing, it's typically because either
# the skia.lib wasn't compiled with Ganesh Vulkan support, or it's in another lib.
# Let's glob everything just to be safe if it's split.
file(GLOB SKIA_DEPS "${SKIA_OUT_DIR}/*.lib")

target_link_libraries(VulkanBasic PRIVATE 
    ${SKIA_LIB} 
    ${SKIA_DEPS}
    ${VULKAN_LIB}
    # Windows system dependencies often needed by Skia
    Opengl32.lib
    Gdi32.lib
    User32.lib
    Advapi32.lib
    Ole32.lib
    OleAut32.lib
    windowscodecs.lib
    Bcrypt.lib
    Dwrite.lib
    Usp10.lib
)

# VulkanBasic.cpp includes some Skia internal test utilities for backend initialization
# like "tools/gpu/vk/VkTestUtils.h" and "src/gpu/...". We need to compile the tools file as well.
# We will check if we need to compile those or if they are in skia.lib in the next step.
# It seems "VulkanBasic.cpp" uses sk_gpu_test::CreateVkBackendContext which is in tools/.

# ==========================================
# wxWidgets Skia Vulkan Demo
# ==========================================
# Try to find wxWidgets. If found, we can build the wxSkiaVulkanDemo
set(wxWidgets_USE_DEBUG ON)
set(wxWidgets_USE_UNICODE ON)
set(wxWidgets_USE_STATIC OFF)  # Force DLL version
find_package(wxWidgets COMPONENTS core base QUIET)
if(wxWidgets_FOUND)
    message(STATUS "wxWidgets found! Building wxSkiaVulkanDemo...")
    message(STATUS "wxWidgets libraries: ${wxWidgets_LIBRARIES}")
    message(STATUS "wxWidgets definitions: ${wxWidgets_DEFINITIONS}")
    include(${wxWidgets_USE_FILE})
    
    add_executable(wxSkiaDemo WIN32 
        wxSkiaVulkanDemo.cpp
        "${SKIA_ROOT}/tools/gpu/vk/VkTestUtils.cpp"
        "${SKIA_ROOT}/tools/gpu/vk/VkTestHelper.cpp"
        "${SKIA_ROOT}/tools/gpu/vk/VkTestMemoryAllocator.cpp"
        "${SKIA_ROOT}/src/gpu/vk/VulkanExtensions.cpp"
        "${SKIA_ROOT}/src/gpu/vk/VulkanPreferredFeatures.cpp"
        "${SKIA_ROOT}/src/gpu/vk/VulkanInterface.cpp"
        "${SKIA_ROOT}/src/gpu/vk/vulkanmemoryallocator/VulkanAMDMemoryAllocator.cpp"
        "${SKIA_ROOT}/third_party/externals/vulkanmemoryallocator/src/VmaUsage.cpp"
    )
    
    # Skia is compiled with /MT (static C-runtime), but OrcaSlicer's wxWidgets uses /MD (DLL C-runtime).
    # We must force wxSkiaDemo to use /MD so it can link against wxWidgets without missing __imp_ symbols.
    set_property(TARGET wxSkiaDemo PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    target_compile_options(wxSkiaDemo PRIVATE "/MD")
    target_compile_definitions(wxSkiaDemo PRIVATE 
        wxDEBUG_LEVEL=0
        __WXMSW__
    )
    
    # Ignore the static runtime library to avoid conflicts
    set_target_properties(wxSkiaDemo PROPERTIES 
        LINK_FLAGS "/NODEFAULTLIB:LIBCMT /NODEFAULTLIB:LIBCMTD"
        WIN32_EXECUTABLE TRUE
    )

    # Needs to link with wxWidgets first, then Skia, Skia dependencies, Vulkan
    target_link_libraries(wxSkiaDemo PRIVATE 
        ${wxWidgets_LIBRARIES}
        ${SKIA_LIB} 
        ${SKIA_DEPS}
        ${VULKAN_LIB}
        # Windows system dependencies
        Opengl32.lib
        Gdi32.lib
        User32.lib
        Advapi32.lib
        Ole32.lib
        OleAut32.lib
        windowscodecs.lib
        Bcrypt.lib
        Dwrite.lib
        Usp10.lib
    )
else()
    message(STATUS "wxWidgets not found. Skipping wxSkiaVulkanDemo. (Set wxWidgets_ROOT_DIR explicitly if needed)")
endif()
