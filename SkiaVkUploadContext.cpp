#include "SkiaVkUploadContext.h"

#include <vector>

namespace skia_port {

namespace {

static void set_error(std::string* error, const std::string& text) {
    if (error) {
        *error = text;
    }
}

static void reset_context(VulkanUploadContext* context) {
    if (!context) {
        return;
    }
    context->instance = VK_NULL_HANDLE;
    context->physicalDevice = VK_NULL_HANDLE;
    context->device = VK_NULL_HANDLE;
    context->queue = VK_NULL_HANDLE;
    context->commandPool = VK_NULL_HANDLE;
}

static uint32_t choose_graphics_queue_family(VkPhysicalDevice physicalDevice, bool* ok) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
    if (count == 0) {
        if (ok) {
            *ok = false;
        }
        return 0;
    }

    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, props.data());

    for (uint32_t i = 0; i < count; ++i) {
        if ((props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0 && props[i].queueCount > 0) {
            if (ok) {
                *ok = true;
            }
            return i;
        }
    }

    if (ok) {
        *ok = false;
    }
    return 0;
}

}  // namespace

bool CreateMinimalVulkanUploadContext(VulkanUploadContext* outContext, std::string* error) {
    if (!outContext) {
        set_error(error, "CreateMinimalVulkanUploadContext: outContext is null");
        return false;
    }
    reset_context(outContext);

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "PathMeshCopiedSkiaDemo";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "PathPatchPrepare";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&instanceInfo, nullptr, &outContext->instance) != VK_SUCCESS) {
        set_error(error, "vkCreateInstance failed");
        return false;
    }

    uint32_t physicalDeviceCount = 0;
    if (vkEnumeratePhysicalDevices(outContext->instance, &physicalDeviceCount, nullptr) !=
            VK_SUCCESS ||
        physicalDeviceCount == 0) {
        set_error(error, "vkEnumeratePhysicalDevices failed or found no devices");
        DestroyVulkanUploadContext(outContext);
        return false;
    }

    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    if (vkEnumeratePhysicalDevices(
            outContext->instance, &physicalDeviceCount, physicalDevices.data()) != VK_SUCCESS) {
        set_error(error, "vkEnumeratePhysicalDevices(list) failed");
        DestroyVulkanUploadContext(outContext);
        return false;
    }
    outContext->physicalDevice = physicalDevices[0];

    bool foundQueueFamily = false;
    const uint32_t queueFamily =
            choose_graphics_queue_family(outContext->physicalDevice, &foundQueueFamily);
    if (!foundQueueFamily) {
        set_error(error, "No compatible graphics queue family");
        DestroyVulkanUploadContext(outContext);
        return false;
    }

    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;

    if (vkCreateDevice(outContext->physicalDevice, &deviceInfo, nullptr, &outContext->device) !=
        VK_SUCCESS) {
        set_error(error, "vkCreateDevice failed");
        DestroyVulkanUploadContext(outContext);
        return false;
    }

    vkGetDeviceQueue(outContext->device, queueFamily, 0, &outContext->queue);
    if (outContext->queue == VK_NULL_HANDLE) {
        set_error(error, "vkGetDeviceQueue returned null");
        DestroyVulkanUploadContext(outContext);
        return false;
    }

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(outContext->device, &poolInfo, nullptr, &outContext->commandPool) !=
        VK_SUCCESS) {
        set_error(error, "vkCreateCommandPool failed");
        DestroyVulkanUploadContext(outContext);
        return false;
    }

    return true;
}

void DestroyVulkanUploadContext(VulkanUploadContext* context) {
    if (!context) {
        return;
    }
    if (context->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(context->device);
    }
    if (context->commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(context->device, context->commandPool, nullptr);
        context->commandPool = VK_NULL_HANDLE;
    }
    if (context->device != VK_NULL_HANDLE) {
        vkDestroyDevice(context->device, nullptr);
        context->device = VK_NULL_HANDLE;
    }
    if (context->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(context->instance, nullptr);
        context->instance = VK_NULL_HANDLE;
    }
    context->physicalDevice = VK_NULL_HANDLE;
    context->queue = VK_NULL_HANDLE;
}

}  // namespace skia_port
