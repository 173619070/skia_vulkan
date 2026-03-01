#include <wx/wx.h>
#include <vector>
#include <algorithm>

#include "include/core/SkCanvas.h"
#include "include/core/SkSurface.h"
#include "include/core/SkRect.h"
#include "include/core/SkPaint.h"
#include "include/core/SkFont.h"
#include "include/core/SkColor.h"
#include "include/core/SkColorSpace.h"
#include "include/core/SkImageInfo.h"
#include "include/gpu/ganesh/GrBackendSurface.h"
#include "include/gpu/ganesh/GrDirectContext.h"
#include "include/gpu/ganesh/SkSurfaceGanesh.h"
#include "include/gpu/ganesh/vk/GrVkDirectContext.h"
#include "include/gpu/ganesh/vk/GrVkBackendSurface.h"
#include "include/gpu/ganesh/vk/GrVkTypes.h"
#include "include/gpu/vk/VulkanBackendContext.h"
#include "include/gpu/vk/VulkanExtensions.h"
#include "include/gpu/vk/VulkanMemoryAllocator.h"

#include "src/gpu/vk/vulkanmemoryallocator/VulkanMemoryAllocatorPriv.h"
#include "src/gpu/GpuTypesPriv.h"
#include "tools/gpu/vk/VkTestUtils.h"

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_win32.h>

#ifdef _WIN32
#include <wx/msw/private.h>
#include <dwmapi.h>
#include <windowsx.h>  // For GET_X_LPARAM, GET_Y_LPARAM
#pragma comment(lib, "dwmapi.lib")
#endif

#define ACQUIRE_INST_VK_PROC(name)                                                           \
    do {                                                                                     \
    fVk##name = reinterpret_cast<PFN_vk##name>(fGetProc("vk" #name, fBackendContext.fInstance, \
                                                       VK_NULL_HANDLE));                     \
    if (fVk##name == nullptr) {                                                              \
        SkDebugf("Function ptr for vk%s could not be acquired\n", #name);                    \
    }                                                                                        \
    } while(false)

// Vulkan-enabled Skia Panel with Swapchain
class SkiaVulkanPanel : public wxPanel {
public:
    SkiaVulkanPanel(wxWindow* parent) 
        : wxPanel(parent, wxID_ANY)
        , fContext(nullptr)
        , fVkDestroyInstance(nullptr)
        , fVkDestroyDebugUtilsMessengerEXT(nullptr)
        , fVkDestroyDevice(nullptr)
        , fVkDestroySurfaceKHR(nullptr)
        , fVkDestroySwapchainKHR(nullptr)
        , fVkGetPhysicalDeviceSurfaceSupportKHR(nullptr)
        , fVkGetPhysicalDeviceSurfaceCapabilitiesKHR(nullptr)
        , fVkGetPhysicalDeviceSurfaceFormatsKHR(nullptr)
        , fVkGetPhysicalDeviceSurfacePresentModesKHR(nullptr)
        , fVkCreateSwapchainKHR(nullptr)
        , fVkGetSwapchainImagesKHR(nullptr)
        , fVkAcquireNextImageKHR(nullptr)
        , fVkQueuePresentKHR(nullptr)
        , fVkCreateSemaphore(nullptr)
        , fVkDestroySemaphore(nullptr)
        , fVkCreateFence(nullptr)
        , fVkDestroyFence(nullptr)
        , fVkWaitForFences(nullptr)
        , fVkResetFences(nullptr)
        , fDebugMessenger(VK_NULL_HANDLE)
        , fSurface(VK_NULL_HANDLE)
        , fSwapchain(VK_NULL_HANDLE)
        , fCommandPool(VK_NULL_HANDLE)
        , fCommandBuffer(VK_NULL_HANDLE)
        , fImageAvailableSemaphore(VK_NULL_HANDLE)
        , fRenderFinishedSemaphore(VK_NULL_HANDLE)
    {
        SetBackgroundStyle(wxBG_STYLE_PAINT);
        Bind(wxEVT_PAINT, &SkiaVulkanPanel::OnPaint, this);
        Bind(wxEVT_SIZE, &SkiaVulkanPanel::OnSize, this);
        
        InitVulkan();
    }

    ~SkiaVulkanPanel() {
        CleanupVulkan();
    }

private:
    sk_sp<GrDirectContext> fContext;
    skgpu::VulkanBackendContext fBackendContext;
    VkDebugUtilsMessengerEXT fDebugMessenger;
    std::unique_ptr<skgpu::VulkanExtensions> fExtensions;
    std::unique_ptr<sk_gpu_test::TestVkFeatures> fFeatures;
    
    // Vulkan function pointers
    skgpu::VulkanGetProc fGetProc;
    PFN_vkDestroyInstance fVkDestroyInstance;
    PFN_vkDestroyDebugUtilsMessengerEXT fVkDestroyDebugUtilsMessengerEXT;
    PFN_vkDestroyDevice fVkDestroyDevice;
    PFN_vkDestroySurfaceKHR fVkDestroySurfaceKHR;
    PFN_vkDestroySwapchainKHR fVkDestroySwapchainKHR;
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR fVkGetPhysicalDeviceSurfaceSupportKHR;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR fVkGetPhysicalDeviceSurfaceCapabilitiesKHR;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR fVkGetPhysicalDeviceSurfaceFormatsKHR;
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR fVkGetPhysicalDeviceSurfacePresentModesKHR;
    PFN_vkCreateSwapchainKHR fVkCreateSwapchainKHR;
    PFN_vkGetSwapchainImagesKHR fVkGetSwapchainImagesKHR;
    PFN_vkAcquireNextImageKHR fVkAcquireNextImageKHR;
    PFN_vkQueuePresentKHR fVkQueuePresentKHR;
    PFN_vkCreateSemaphore fVkCreateSemaphore;
    PFN_vkDestroySemaphore fVkDestroySemaphore;
    PFN_vkCreateFence fVkCreateFence;
    PFN_vkDestroyFence fVkDestroyFence;
    PFN_vkWaitForFences fVkWaitForFences;
    PFN_vkResetFences fVkResetFences;
    PFN_vkCreateCommandPool fVkCreateCommandPool;
    PFN_vkDestroyCommandPool fVkDestroyCommandPool;
    PFN_vkAllocateCommandBuffers fVkAllocateCommandBuffers;
    PFN_vkBeginCommandBuffer fVkBeginCommandBuffer;
    PFN_vkEndCommandBuffer fVkEndCommandBuffer;
    PFN_vkCmdPipelineBarrier fVkCmdPipelineBarrier;
    PFN_vkCmdBlitImage fVkCmdBlitImage;
    PFN_vkQueueSubmit fVkQueueSubmit;
    
    // Swapchain resources
    VkSurfaceKHR fSurface;
    VkSwapchainKHR fSwapchain;
    VkFormat fSwapchainFormat;
    VkExtent2D fSwapchainExtent;
    std::vector<VkImage> fSwapchainImages;
    sk_sp<SkSurface> fRenderSurface;  // Single offscreen render surface
    
    // Command buffer for blit operations
    VkCommandPool fCommandPool;
    VkCommandBuffer fCommandBuffer;
    
    // Synchronization
    VkSemaphore fImageAvailableSemaphore;
    VkSemaphore fRenderFinishedSemaphore;

    bool InitVulkan() {
        fExtensions = std::make_unique<skgpu::VulkanExtensions>();
        fFeatures = std::make_unique<sk_gpu_test::TestVkFeatures>();

        // Load Vulkan library and create instance/device
        PFN_vkGetInstanceProcAddr instProc;
        if (!sk_gpu_test::LoadVkLibraryAndGetProcAddrFuncs(&instProc)) {
            wxLogError("Failed to load Vulkan library");
            return false;
        }

        fBackendContext.fInstance = VK_NULL_HANDLE;
        fBackendContext.fDevice = VK_NULL_HANDLE;

        if (!sk_gpu_test::CreateVkBackendContext(
                    instProc, &fBackendContext, fExtensions.get(), fFeatures.get(), &fDebugMessenger)) {
            wxLogError("Failed to create Vulkan backend context");
            return false;
        }

        fGetProc = fBackendContext.fGetProc;
        ACQUIRE_INST_VK_PROC(DestroyInstance);
        if (fDebugMessenger != VK_NULL_HANDLE) {
            ACQUIRE_INST_VK_PROC(DestroyDebugUtilsMessengerEXT);
        }
        ACQUIRE_INST_VK_PROC(DestroyDevice);
        ACQUIRE_INST_VK_PROC(DestroySurfaceKHR);
        ACQUIRE_INST_VK_PROC(DestroySwapchainKHR);
        ACQUIRE_INST_VK_PROC(GetPhysicalDeviceSurfaceSupportKHR);
        ACQUIRE_INST_VK_PROC(GetPhysicalDeviceSurfaceCapabilitiesKHR);
        ACQUIRE_INST_VK_PROC(GetPhysicalDeviceSurfaceFormatsKHR);
        ACQUIRE_INST_VK_PROC(GetPhysicalDeviceSurfacePresentModesKHR);
        ACQUIRE_INST_VK_PROC(CreateSwapchainKHR);
        ACQUIRE_INST_VK_PROC(GetSwapchainImagesKHR);
        ACQUIRE_INST_VK_PROC(AcquireNextImageKHR);
        ACQUIRE_INST_VK_PROC(QueuePresentKHR);
        ACQUIRE_INST_VK_PROC(CreateSemaphore);
        ACQUIRE_INST_VK_PROC(DestroySemaphore);
        ACQUIRE_INST_VK_PROC(CreateFence);
        ACQUIRE_INST_VK_PROC(DestroyFence);
        ACQUIRE_INST_VK_PROC(WaitForFences);
        ACQUIRE_INST_VK_PROC(ResetFences);
        ACQUIRE_INST_VK_PROC(CreateCommandPool);
        ACQUIRE_INST_VK_PROC(DestroyCommandPool);
        ACQUIRE_INST_VK_PROC(AllocateCommandBuffers);
        ACQUIRE_INST_VK_PROC(BeginCommandBuffer);
        ACQUIRE_INST_VK_PROC(EndCommandBuffer);
        ACQUIRE_INST_VK_PROC(CmdPipelineBarrier);
        ACQUIRE_INST_VK_PROC(CmdBlitImage);
        ACQUIRE_INST_VK_PROC(QueueSubmit);

        fBackendContext.fMemoryAllocator = skgpu::VulkanMemoryAllocators::Make(
                fBackendContext, skgpu::ThreadSafe::kNo);

        // Create GrDirectContext
        fContext = GrDirectContexts::MakeVulkan(fBackendContext);
        if (!fContext) {
            wxLogError("Failed to create Vulkan GrDirectContext");
            CleanupVulkan();
            return false;
        }

        // Create Win32 surface
        if (!CreateSurface()) {
            wxLogError("Failed to create Vulkan surface");
            CleanupVulkan();
            return false;
        }

        // Create synchronization objects
        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        
        if (fVkCreateSemaphore(fBackendContext.fDevice, &semaphoreInfo, nullptr, &fImageAvailableSemaphore) != VK_SUCCESS ||
            fVkCreateSemaphore(fBackendContext.fDevice, &semaphoreInfo, nullptr, &fRenderFinishedSemaphore) != VK_SUCCESS) {
            wxLogError("Failed to create semaphores");
            CleanupVulkan();
            return false;
        }

        // Create command pool
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = fBackendContext.fGraphicsQueueIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        
        if (fVkCreateCommandPool(fBackendContext.fDevice, &poolInfo, nullptr, &fCommandPool) != VK_SUCCESS) {
            wxLogError("Failed to create command pool");
            CleanupVulkan();
            return false;
        }

        // Allocate command buffer
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = fCommandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        
        if (fVkAllocateCommandBuffers(fBackendContext.fDevice, &allocInfo, &fCommandBuffer) != VK_SUCCESS) {
            wxLogError("Failed to allocate command buffer");
            CleanupVulkan();
            return false;
        }

        return true;
    }

    bool CreateSurface() {
#ifdef _WIN32
        HWND hwnd = GetHWND();
        if (!hwnd) {
            wxLogError("Failed to get HWND");
            return false;
        }

        VkWin32SurfaceCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        createInfo.hwnd = hwnd;
        createInfo.hinstance = GetModuleHandle(nullptr);

        PFN_vkCreateWin32SurfaceKHR vkCreateWin32SurfaceKHR = 
            reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
                fGetProc("vkCreateWin32SurfaceKHR", fBackendContext.fInstance, VK_NULL_HANDLE));
        
        if (!vkCreateWin32SurfaceKHR) {
            wxLogError("Failed to get vkCreateWin32SurfaceKHR");
            return false;
        }

        if (vkCreateWin32SurfaceKHR(fBackendContext.fInstance, &createInfo, nullptr, &fSurface) != VK_SUCCESS) {
            wxLogError("Failed to create Win32 surface");
            return false;
        }

        // Verify surface support
        VkBool32 supported = false;
        fVkGetPhysicalDeviceSurfaceSupportKHR(fBackendContext.fPhysicalDevice, 
                                              fBackendContext.fGraphicsQueueIndex, 
                                              fSurface, &supported);
        if (!supported) {
            wxLogError("Surface not supported by physical device");
            return false;
        }

        return true;
#else
        wxLogError("Platform not supported");
        return false;
#endif
    }

    bool CreateSwapchain(int width, int height) {
        if (width <= 0 || height <= 0) return false;

        wxLogMessage("Creating swapchain: %dx%d", width, height);

        // Get surface capabilities
        VkSurfaceCapabilitiesKHR capabilities;
        fVkGetPhysicalDeviceSurfaceCapabilitiesKHR(fBackendContext.fPhysicalDevice, fSurface, &capabilities);

        wxLogMessage("Surface capabilities: min=%dx%d, max=%dx%d, current=%dx%d",
                     capabilities.minImageExtent.width, capabilities.minImageExtent.height,
                     capabilities.maxImageExtent.width, capabilities.maxImageExtent.height,
                     capabilities.currentExtent.width, capabilities.currentExtent.height);

        // Choose surface format
        uint32_t formatCount;
        fVkGetPhysicalDeviceSurfaceFormatsKHR(fBackendContext.fPhysicalDevice, fSurface, &formatCount, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        fVkGetPhysicalDeviceSurfaceFormatsKHR(fBackendContext.fPhysicalDevice, fSurface, &formatCount, formats.data());

        VkSurfaceFormatKHR surfaceFormat = formats[0];
        for (const auto& format : formats) {
            if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                surfaceFormat = format;
                break;
            }
        }
        fSwapchainFormat = surfaceFormat.format;

        // Choose present mode
        uint32_t presentModeCount;
        fVkGetPhysicalDeviceSurfacePresentModesKHR(fBackendContext.fPhysicalDevice, fSurface, &presentModeCount, nullptr);
        std::vector<VkPresentModeKHR> presentModes(presentModeCount);
        fVkGetPhysicalDeviceSurfacePresentModesKHR(fBackendContext.fPhysicalDevice, fSurface, &presentModeCount, presentModes.data());

        VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
        for (const auto& mode : presentModes) {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
                presentMode = mode;
                break;
            }
        }

        // Choose extent
        fSwapchainExtent = capabilities.currentExtent;
        if (fSwapchainExtent.width == UINT32_MAX) {
            fSwapchainExtent.width = std::clamp(static_cast<uint32_t>(width), 
                                                 capabilities.minImageExtent.width, 
                                                 capabilities.maxImageExtent.width);
            fSwapchainExtent.height = std::clamp(static_cast<uint32_t>(height), 
                                                  capabilities.minImageExtent.height, 
                                                  capabilities.maxImageExtent.height);
        }

        uint32_t imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
            imageCount = capabilities.maxImageCount;
        }

        // Create swapchain
        VkSwapchainCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = fSurface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = fSwapchainExtent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.preTransform = capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (fVkCreateSwapchainKHR(fBackendContext.fDevice, &createInfo, nullptr, &fSwapchain) != VK_SUCCESS) {
            wxLogError("Failed to create swapchain");
            return false;
        }

        // Get swapchain images
        fVkGetSwapchainImagesKHR(fBackendContext.fDevice, fSwapchain, &imageCount, nullptr);
        fSwapchainImages.resize(imageCount);
        fVkGetSwapchainImagesKHR(fBackendContext.fDevice, fSwapchain, &imageCount, fSwapchainImages.data());

        // Create Skia surface for offscreen rendering
        SkImageInfo imageInfo = SkImageInfo::Make(
            fSwapchainExtent.width, fSwapchainExtent.height,
            kBGRA_8888_SkColorType, kPremul_SkAlphaType);
        
        fRenderSurface = SkSurfaces::RenderTarget(
            fContext.get(), skgpu::Budgeted::kYes, imageInfo);
        
        if (!fRenderSurface) {
            wxLogError("Failed to create offscreen render surface");
            return false;
        }
        
        wxLogMessage("Successfully created offscreen Skia surface for %zu swapchain images", 
                     fSwapchainImages.size());

        return true;
    }

    void DestroySwapchain() {
        fRenderSurface.reset();
        
        if (fSwapchain != VK_NULL_HANDLE) {
            fVkDestroySwapchainKHR(fBackendContext.fDevice, fSwapchain, nullptr);
            fSwapchain = VK_NULL_HANDLE;
        }
        
        fSwapchainImages.clear();
    }

    void CleanupVulkan() {
        // Wait for device to be idle
        if (fBackendContext.fDevice) {
            PFN_vkDeviceWaitIdle vkDeviceWaitIdle = 
                reinterpret_cast<PFN_vkDeviceWaitIdle>(
                    fGetProc("vkDeviceWaitIdle", VK_NULL_HANDLE, fBackendContext.fDevice));
            if (vkDeviceWaitIdle) {
                vkDeviceWaitIdle(fBackendContext.fDevice);
            }
        }

        // Destroy swapchain
        DestroySwapchain();

        // Destroy command pool (this also frees command buffers)
        if (fCommandPool != VK_NULL_HANDLE) {
            fVkDestroyCommandPool(fBackendContext.fDevice, fCommandPool, nullptr);
            fCommandPool = VK_NULL_HANDLE;
            fCommandBuffer = VK_NULL_HANDLE;
        }

        // Destroy synchronization objects
        if (fImageAvailableSemaphore != VK_NULL_HANDLE) {
            fVkDestroySemaphore(fBackendContext.fDevice, fImageAvailableSemaphore, nullptr);
            fImageAvailableSemaphore = VK_NULL_HANDLE;
        }
        if (fRenderFinishedSemaphore != VK_NULL_HANDLE) {
            fVkDestroySemaphore(fBackendContext.fDevice, fRenderFinishedSemaphore, nullptr);
            fRenderFinishedSemaphore = VK_NULL_HANDLE;
        }

        // CRITICAL: Must release context before destroying device
        fContext.reset();
        
        // Release the memory allocator explicitly
        fBackendContext.fMemoryAllocator.reset();

        // Destroy surface
        if (fSurface != VK_NULL_HANDLE) {
            fVkDestroySurfaceKHR(fBackendContext.fInstance, fSurface, nullptr);
            fSurface = VK_NULL_HANDLE;
        }

        if (fVkDestroyDevice && fBackendContext.fDevice) {
            fVkDestroyDevice(fBackendContext.fDevice, nullptr);
            fBackendContext.fDevice = VK_NULL_HANDLE;
        }
        if (fVkDestroyDebugUtilsMessengerEXT && fDebugMessenger != VK_NULL_HANDLE) {
            fVkDestroyDebugUtilsMessengerEXT(fBackendContext.fInstance, fDebugMessenger, nullptr);
            fDebugMessenger = VK_NULL_HANDLE;
        }
        if (fVkDestroyInstance && fBackendContext.fInstance) {
            fVkDestroyInstance(fBackendContext.fInstance, nullptr);
            fBackendContext.fInstance = VK_NULL_HANDLE;
        }
    }

    void OnSize(wxSizeEvent& event) {
        if (fContext && fSurface != VK_NULL_HANDLE) {
            wxSize size = GetClientSize();
            if (size.x > 0 && size.y > 0) {
                // Recreate swapchain on resize
                DestroySwapchain();
                CreateSwapchain(size.x, size.y);
            }
        }
        Refresh();
        event.Skip();
    }

    void OnPaint(wxPaintEvent& event) {
        if (!fContext || fSwapchain == VK_NULL_HANDLE) {
            // Need to create swapchain on first paint
            wxSize size = GetClientSize();
            if (size.x > 0 && size.y > 0 && fSurface != VK_NULL_HANDLE) {
                CreateSwapchain(size.x, size.y);
            }
            
            if (fSwapchain == VK_NULL_HANDLE) {
                return;
            }
        }

        // Acquire next image from swapchain
        uint32_t imageIndex;
        VkResult result = fVkAcquireNextImageKHR(fBackendContext.fDevice, fSwapchain, 
                                                  UINT64_MAX, fImageAvailableSemaphore, 
                                                  VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            wxSize size = GetClientSize();
            DestroySwapchain();
            CreateSwapchain(size.x, size.y);
            return;
        } else if (result != VK_SUCCESS) {
            wxLogError("Failed to acquire swapchain image");
            return;
        }

        // Get the offscreen Skia surface
        if (!fRenderSurface) {
            wxLogError("Invalid Skia surface");
            return;
        }

        SkCanvas* canvas = fRenderSurface->getCanvas();
        
        // Draw content to offscreen surface
        DrawContent(canvas, fSwapchainExtent.width, fSwapchainExtent.height);

        // Flush Skia rendering
        fContext->flush(fRenderSurface.get());
        fContext->submit();

        // Get the VkImage from Skia surface
        GrBackendRenderTarget backendRT = SkSurfaces::GetBackendRenderTarget(
            fRenderSurface.get(), SkSurfaces::BackendHandleAccess::kFlushRead);
        
        if (!backendRT.isValid()) {
            wxLogError("Failed to get backend render target");
            return;
        }

        GrVkImageInfo imageInfo;
        if (!GrBackendRenderTargets::GetVkImageInfo(backendRT, &imageInfo)) {
            wxLogError("Failed to get Vulkan image info");
            return;
        }

        VkImage srcImage = imageInfo.fImage;
        VkImage dstImage = fSwapchainImages[imageIndex];

        // Record blit command
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        
        fVkBeginCommandBuffer(fCommandBuffer, &beginInfo);

        // Transition source image to TRANSFER_SRC_OPTIMAL
        VkImageMemoryBarrier srcBarrier = {};
        srcBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        srcBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        srcBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        srcBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        srcBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        srcBarrier.image = srcImage;
        srcBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        srcBarrier.subresourceRange.baseMipLevel = 0;
        srcBarrier.subresourceRange.levelCount = 1;
        srcBarrier.subresourceRange.baseArrayLayer = 0;
        srcBarrier.subresourceRange.layerCount = 1;
        srcBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        srcBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        fVkCmdPipelineBarrier(fCommandBuffer,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &srcBarrier);

        // Transition destination image to TRANSFER_DST_OPTIMAL
        VkImageMemoryBarrier dstBarrier = {};
        dstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        dstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        dstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        dstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        dstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        dstBarrier.image = dstImage;
        dstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        dstBarrier.subresourceRange.baseMipLevel = 0;
        dstBarrier.subresourceRange.levelCount = 1;
        dstBarrier.subresourceRange.baseArrayLayer = 0;
        dstBarrier.subresourceRange.layerCount = 1;
        dstBarrier.srcAccessMask = 0;
        dstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        fVkCmdPipelineBarrier(fCommandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &dstBarrier);

        // Blit image
        VkImageBlit blitRegion = {};
        blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blitRegion.srcSubresource.layerCount = 1;
        blitRegion.srcOffsets[1].x = fSwapchainExtent.width;
        blitRegion.srcOffsets[1].y = fSwapchainExtent.height;
        blitRegion.srcOffsets[1].z = 1;
        blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blitRegion.dstSubresource.layerCount = 1;
        blitRegion.dstOffsets[1].x = fSwapchainExtent.width;
        blitRegion.dstOffsets[1].y = fSwapchainExtent.height;
        blitRegion.dstOffsets[1].z = 1;

        fVkCmdBlitImage(fCommandBuffer,
            srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blitRegion, VK_FILTER_NEAREST);

        // Transition source back to COLOR_ATTACHMENT_OPTIMAL
        srcBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        srcBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        srcBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        srcBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        fVkCmdPipelineBarrier(fCommandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, nullptr, 0, nullptr, 1, &srcBarrier);

        // Transition destination to PRESENT_SRC
        dstBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        dstBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        dstBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dstBarrier.dstAccessMask = 0;

        fVkCmdPipelineBarrier(fCommandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &dstBarrier);

        fVkEndCommandBuffer(fCommandBuffer);

        // Submit command buffer
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &fCommandBuffer;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &fImageAvailableSemaphore;
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_TRANSFER_BIT};
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &fRenderFinishedSemaphore;

        if (fVkQueueSubmit(fBackendContext.fQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            wxLogError("Failed to submit command buffer");
            return;
        }

        // Present
        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &fRenderFinishedSemaphore;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &fSwapchain;
        presentInfo.pImageIndices = &imageIndex;

        result = fVkQueuePresentKHR(fBackendContext.fQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            wxSize size = GetClientSize();
            DestroySwapchain();
            CreateSwapchain(size.x, size.y);
        }
    }

    void DrawContent(SkCanvas* canvas, int width, int height) {
        // Clear background
        canvas->clear(SK_ColorBLACK);

        // Draw filled blue rectangle
        SkPaint paint;
        paint.setAntiAlias(true);
        paint.setColor(SK_ColorBLUE);
        SkRect rect = SkRect::MakeXYWH(50, 50, 300, 150);
        canvas->drawRect(rect, paint);
        
        // Draw stroked red rectangle
        paint.setStyle(SkPaint::kStroke_Style);
        paint.setColor(SK_ColorRED);
        paint.setStrokeWidth(5);
        canvas->drawRect(rect, paint);

        // Draw text
        SkFont font(nullptr, 32);
        paint.setStyle(SkPaint::kFill_Style);
        paint.setColor(SK_ColorBLACK);
        canvas->drawString("Vulkan Direct Rendering with Swapchain!", 50, 250, font, paint);
        
        // Draw green circle
        paint.setColor(0xFF009900);
        canvas->drawCircle(400, 400, 80, paint);

        // Draw Vulkan indicator
        paint.setColor(SK_ColorMAGENTA);
        font.setSize(24);
        canvas->drawString("Direct Vulkan Swapchain Rendering", 50, height - 30, font, paint);
    }
};

class MainFrame : public wxFrame {
public:
    MainFrame(const wxString& title) 
        : wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, wxSize(800, 600),
                  wxDEFAULT_FRAME_STYLE)  // Keep default style first
        , m_scale_factor(1.0f)
        , m_dragging(false)
    {
        SetBackgroundColour(wxColour(45, 45, 48));  // Dark background
        
#ifdef _WIN32
        // Remove Windows title bar and enable custom drawing
        HWND hwnd = GetHWND();
        
        // Check if DWM composition is enabled
        BOOL compositionEnabled = FALSE;
        DwmIsCompositionEnabled(&compositionEnabled);
        
        LONG oldStyle = WS_OVERLAPPEDWINDOW | WS_THICKFRAME | WS_CAPTION | WS_SYSMENU | WS_MAXIMIZEBOX | WS_MINIMIZEBOX;
        LONG newStyle = WS_POPUP | WS_THICKFRAME | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX;
        
        if (compositionEnabled) {
            newStyle |= WS_CAPTION;
        }
        
        LONG currentStyle = GetWindowLong(hwnd, GWL_STYLE);
        SetWindowLong(hwnd, GWL_STYLE, (currentStyle & ~oldStyle) | newStyle);
        
        // Force window to update
        SetWindowPos(hwnd, NULL, 0, 0, 0, 0,
                     SWP_FRAMECHANGED | SWP_NOACTIVATE | SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER);
        
        // Extend frame into client area to remove white border
        UpdateFrameMargins();
#endif
        
        // Create main sizer
        wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);
        
        // Create custom title bar
        CreateTitleBar();
        mainSizer->Add(m_titleBar, 0, wxEXPAND);
        
        // Add the Skia panel
        SkiaVulkanPanel* skiaPanel = new SkiaVulkanPanel(this);
        mainSizer->Add(skiaPanel, 1, wxEXPAND);
        
        SetSizer(mainSizer);
        SetMinSize(wxSize(640, 480));
        Centre();
        
        // Bind DPI change event
        Bind(wxEVT_DPI_CHANGED, &MainFrame::OnDPIChanged, this);
        
        // Bind size event to update margins on maximize/restore
        Bind(wxEVT_SIZE, &MainFrame::OnFrameSize, this);
    }

#ifdef _WIN32
protected:
    void UpdateFrameMargins() {
        HWND hwnd = GetHWND();
        BOOL compositionEnabled = FALSE;
        DwmIsCompositionEnabled(&compositionEnabled);
        
        if (compositionEnabled) {
            // Use 0 margins when maximized to eliminate border, 1 pixel shadow when normal
            MARGINS margins = ::IsZoomed(hwnd) ? MARGINS{0, 0, 0, 0} : MARGINS{1, 1, 1, 1};
            DwmExtendFrameIntoClientArea(hwnd, &margins);
        }
    }
    
    void OnFrameSize(wxSizeEvent& event) {
        UpdateFrameMargins();
        event.Skip();
    }

    WXLRESULT MSWWindowProc(WXUINT nMsg, WXWPARAM wParam, WXLPARAM lParam) override {
        HWND hWnd = GetHWND();
        
        switch (nMsg) {
        case WM_NCACTIVATE: {
            // Disable non-client area painting
            lParam = -1;
            break;
        }
        
        case WM_NCCALCSIZE: {
            if (static_cast<BOOL>(wParam) == FALSE) {
                return 0;
            }
            
            // When wParam is TRUE, we need to adjust the client area
            NCCALCSIZE_PARAMS* params = reinterpret_cast<NCCALCSIZE_PARAMS*>(lParam);
            
            // Check if window is maximized
            const bool isMaximized = ::IsZoomed(hWnd);
            
            if (isMaximized) {
                // When maximized, adjust to work area to prevent overlap with taskbar
                HMONITOR hMonitor = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
                MONITORINFO monitorInfo;
                monitorInfo.cbSize = sizeof(MONITORINFO);
                if (GetMonitorInfo(hMonitor, &monitorInfo)) {
                    params->rgrc[0] = monitorInfo.rcWork;
                }
            }
            // When not maximized, don't modify - let the entire window be client area
            
            return 0;
        }
        
        // Block these undocumented messages that draw themed window borders
        case 0xAE:  // WM_NCUAHDRAWCAPTION
        case 0xAF:  // WM_NCUAHDRAWFRAME
            return 0;
        
        case WM_NCPAINT: {
            // Block WM_NCPAINT when DWM composition is disabled
            BOOL compositionEnabled = FALSE;
            DwmIsCompositionEnabled(&compositionEnabled);
            if (!compositionEnabled) {
                return 0;
            }
            break;
        }
        
        case WM_NCHITTEST: {
            // Get mouse position in screen and client coordinates
            POINT screenPt;
            screenPt.x = GET_X_LPARAM(lParam);
            screenPt.y = GET_Y_LPARAM(lParam);
            
            POINT clientPt = screenPt;
            ::ScreenToClient(hWnd, &clientPt);
            
            RECT clientRect;
            ::GetClientRect(hWnd, &clientRect);
            const LONG windowWidth = clientRect.right - clientRect.left;
            const LONG windowHeight = clientRect.bottom - clientRect.top;
            
            const bool isMax = ::IsZoomed(hWnd);
            const int resizeBorderThickness = 8;  // Border thickness for resizing
            const int titleBarHeight = m_titleBar ? m_titleBar->GetSize().GetHeight() : ScaleValue(32);
            
            // Check if mouse is over a control button (minimize, maximize, close)
            bool isOverControl = false;
            if (m_titleBar) {
                wxPoint wxScreenPt(screenPt.x, screenPt.y);
                if (m_minimizeBtn && m_minimizeBtn->GetScreenRect().Contains(wxScreenPt)) isOverControl = true;
                if (m_maximizeBtn && m_maximizeBtn->GetScreenRect().Contains(wxScreenPt)) isOverControl = true;
                if (m_closeBtn && m_closeBtn->GetScreenRect().Contains(wxScreenPt)) isOverControl = true;
            }
            
            // Determine if in title bar area
            bool isTitleBar = false;
            if (isMax) {
                // When maximized, title bar is from top to titleBarHeight
                isTitleBar = (clientPt.y >= 0) && (clientPt.y <= titleBarHeight) &&
                            (clientPt.x >= 0) && (clientPt.x <= windowWidth) &&
                            !isOverControl;
            } else {
                // When normal, title bar is from resizeBorderThickness to titleBarHeight
                isTitleBar = (clientPt.y > resizeBorderThickness) && (clientPt.y <= titleBarHeight) &&
                            (clientPt.x > resizeBorderThickness) && (clientPt.x < (windowWidth - resizeBorderThickness)) &&
                            !isOverControl;
            }
            
            // Check if mouse button is pressed (for dragging)
            const bool mousePressed = (GetSystemMetrics(SM_SWAPBUTTON) ? 
                                      GetAsyncKeyState(VK_RBUTTON) : 
                                      GetAsyncKeyState(VK_LBUTTON)) < 0;
            
            // When maximized, only allow title bar dragging
            if (isMax) {
                if (isTitleBar && mousePressed) {
                    return HTCAPTION;
                }
                return HTCLIENT;
            }
            
            // When not maximized, check borders for resizing
            const bool isTop = clientPt.y <= resizeBorderThickness;
            const bool isBottom = clientPt.y >= (windowHeight - resizeBorderThickness);
            const bool isLeft = clientPt.x <= resizeBorderThickness;
            const bool isRight = clientPt.x >= (windowWidth - resizeBorderThickness);
            
            // Check corners first
            if (isTop && isLeft) return HTTOPLEFT;
            if (isTop && isRight) return HTTOPRIGHT;
            if (isBottom && isLeft) return HTBOTTOMLEFT;
            if (isBottom && isRight) return HTBOTTOMRIGHT;
            
            // Check edges
            if (isTop) return HTTOP;
            if (isBottom) return HTBOTTOM;
            if (isLeft) return HTLEFT;
            if (isRight) return HTRIGHT;
            
            // Check title bar for dragging
            if (isTitleBar && mousePressed) {
                return HTCAPTION;
            }
            
            return HTCLIENT;
        }
        }
        
        return wxFrame::MSWWindowProc(nMsg, wParam, lParam);
    }
#endif

private:
    wxPanel* m_titleBar;
    wxStaticText* m_titleText;
    wxButton* m_minimizeBtn;
    wxButton* m_maximizeBtn;
    wxButton* m_closeBtn;
    float m_scale_factor;
    bool m_dragging;
    wxPoint m_dragStart;
    
    void CreateTitleBar() {
        m_titleBar = new wxPanel(this, wxID_ANY);
        m_titleBar->SetBackgroundColour(wxColour(30, 30, 30));  // Darker title bar
        m_titleBar->SetMinSize(wxSize(-1, ScaleValue(32)));
        
        wxBoxSizer* titleSizer = new wxBoxSizer(wxHORIZONTAL);
        
        // Title text
        m_titleText = new wxStaticText(m_titleBar, wxID_ANY, "Skia Vulkan Demo");
        m_titleText->SetForegroundColour(*wxWHITE);
        wxFont titleFont = m_titleText->GetFont();
        titleFont.SetPointSize(ScaleValue(10));
        m_titleText->SetFont(titleFont);
        titleSizer->Add(m_titleText, 1, wxALIGN_CENTER_VERTICAL | wxLEFT, ScaleValue(10));
        
        // Window control buttons
        int btnSize = ScaleValue(32);
        
        m_minimizeBtn = CreateTitleButton(m_titleBar, "_", btnSize);
        m_maximizeBtn = CreateTitleButton(m_titleBar, "□", btnSize);
        m_closeBtn = CreateTitleButton(m_titleBar, "×", btnSize);
        
        titleSizer->Add(m_minimizeBtn, 0, wxALIGN_CENTER_VERTICAL);
        titleSizer->Add(m_maximizeBtn, 0, wxALIGN_CENTER_VERTICAL);
        titleSizer->Add(m_closeBtn, 0, wxALIGN_CENTER_VERTICAL);
        
        m_titleBar->SetSizer(titleSizer);
        
        // Bind events for dragging
        m_titleBar->Bind(wxEVT_LEFT_DOWN, &MainFrame::OnTitleBarLeftDown, this);
        m_titleBar->Bind(wxEVT_LEFT_UP, &MainFrame::OnTitleBarLeftUp, this);
        m_titleBar->Bind(wxEVT_MOTION, &MainFrame::OnTitleBarMotion, this);
        m_titleBar->Bind(wxEVT_LEFT_DCLICK, &MainFrame::OnTitleBarDoubleClick, this);
        m_titleText->Bind(wxEVT_LEFT_DOWN, &MainFrame::OnTitleBarLeftDown, this);
        m_titleText->Bind(wxEVT_LEFT_UP, &MainFrame::OnTitleBarLeftUp, this);
        m_titleText->Bind(wxEVT_MOTION, &MainFrame::OnTitleBarMotion, this);
        m_titleText->Bind(wxEVT_LEFT_DCLICK, &MainFrame::OnTitleBarDoubleClick, this);
        
        // Bind button events
        m_minimizeBtn->Bind(wxEVT_BUTTON, &MainFrame::OnMinimize, this);
        m_maximizeBtn->Bind(wxEVT_BUTTON, &MainFrame::OnMaximize, this);
        m_closeBtn->Bind(wxEVT_BUTTON, &MainFrame::OnClose, this);
    }
    
    wxButton* CreateTitleButton(wxWindow* parent, const wxString& label, int size) {
        wxButton* btn = new wxButton(parent, wxID_ANY, label, 
                                     wxDefaultPosition, wxSize(size, size),
                                     wxBORDER_NONE);
        btn->SetBackgroundColour(wxColour(30, 30, 30));
        btn->SetForegroundColour(*wxWHITE);
        
        // Hover effects
        btn->Bind(wxEVT_ENTER_WINDOW, [btn, label](wxMouseEvent& evt) {
            if (label == "×") {
                btn->SetBackgroundColour(wxColour(232, 17, 35));  // Red for close
            } else {
                btn->SetBackgroundColour(wxColour(60, 60, 60));
            }
            btn->Refresh();
        });
        
        btn->Bind(wxEVT_LEAVE_WINDOW, [btn](wxMouseEvent& evt) {
            btn->SetBackgroundColour(wxColour(30, 30, 30));
            btn->Refresh();
        });
        
        return btn;
    }
    
    int ScaleValue(int value) const {
        return static_cast<int>(value * m_scale_factor);
    }
    
    void OnDPIChanged(wxDPIChangedEvent& event) {
        wxSize dpi = event.GetNewDPI();
        m_scale_factor = dpi.GetWidth() / 96.0f;  // 96 DPI is standard
        
        // Update title bar height
        m_titleBar->SetMinSize(wxSize(-1, ScaleValue(32)));
        
        // Update title font
        wxFont titleFont = m_titleText->GetFont();
        titleFont.SetPointSize(ScaleValue(10));
        m_titleText->SetFont(titleFont);
        
        // Update button sizes
        int btnSize = ScaleValue(32);
        m_minimizeBtn->SetSize(btnSize, btnSize);
        m_maximizeBtn->SetSize(btnSize, btnSize);
        m_closeBtn->SetSize(btnSize, btnSize);
        
        Layout();
        Refresh();
    }
    
    void OnTitleBarLeftDown(wxMouseEvent& event) {
        m_dragging = true;
        wxPoint clientPos = event.GetPosition();
        // Convert to screen coordinates relative to the title bar
        m_dragStart = m_titleBar->ClientToScreen(clientPos) - GetPosition();
        m_titleBar->CaptureMouse();
    }
    
    void OnTitleBarLeftUp(wxMouseEvent& event) {
        if (m_dragging) {
            m_dragging = false;
            if (m_titleBar->HasCapture()) {
                m_titleBar->ReleaseMouse();
            }
        }
    }
    
    void OnTitleBarMotion(wxMouseEvent& event) {
        if (m_dragging && event.Dragging()) {
            wxPoint screenPos = m_titleBar->ClientToScreen(event.GetPosition());
            Move(screenPos - m_dragStart);
        }
    }
    
    void OnTitleBarDoubleClick(wxMouseEvent& event) {
        HWND hwnd = GetHWND();
        if (::IsZoomed(hwnd)) {
            Restore();
            m_maximizeBtn->SetLabel("□");
        } else {
            Maximize();
            m_maximizeBtn->SetLabel("❐");
        }
    }
    
    void OnMinimize(wxCommandEvent& event) {
        Iconize(true);
    }
    
    void OnMaximize(wxCommandEvent& event) {
        HWND hwnd = GetHWND();
        if (::IsZoomed(hwnd)) {
            Restore();
            m_maximizeBtn->SetLabel("□");
        } else {
            Maximize();
            m_maximizeBtn->SetLabel("❐");
        }
    }
    
    void OnClose(wxCommandEvent& event) {
        Close(true);
    }
};

class MyApp : public wxApp {
public:
    virtual bool OnInit() {
        if (!wxApp::OnInit()) return false;
        
        MainFrame* frame = new MainFrame("Skia Vulkan with wxWidgets Demo");
        frame->Show(true);
        return true;
    }
};

wxIMPLEMENT_APP(MyApp);
