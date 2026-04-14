#pragma once

#include <vulkan/vulkan.h>

#include <string>

namespace skia_port {

struct VulkanUploadContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
};

bool CreateMinimalVulkanUploadContext(VulkanUploadContext* outContext,
                                      std::string* error = nullptr);
void DestroyVulkanUploadContext(VulkanUploadContext* context);

}  // namespace skia_port
