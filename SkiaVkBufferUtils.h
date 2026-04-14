#pragma once

#include "SkiaVkUploadContext.h"

#include <cstddef>
#include <string>

namespace skia_port {

struct VkOwnedBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t sizeBytes = 0;
};

uint32_t FindMemoryType(const VulkanUploadContext& ctx,
                        uint32_t typeBits,
                        VkMemoryPropertyFlags props,
                        std::string* error = nullptr);

bool CreateOwnedBuffer(const VulkanUploadContext& ctx,
                       VkDeviceSize size,
                       VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags props,
                       VkOwnedBuffer* out,
                       std::string* error = nullptr);

bool UploadHostVisibleBuffer(const VulkanUploadContext& ctx,
                             const void* src,
                             size_t size,
                             VkOwnedBuffer* dst,
                             std::string* error = nullptr);

bool CreateDeviceLocalBufferWithData(const VulkanUploadContext& ctx,
                                     const void* src,
                                     size_t size,
                                     VkBufferUsageFlags usage,
                                     VkOwnedBuffer* out,
                                     std::string* error = nullptr);

void DestroyOwnedBuffer(const VulkanUploadContext& ctx, VkOwnedBuffer* buffer);

VkFormat FindDepthStencilFormat(VkPhysicalDevice physicalDevice);

}  // namespace skia_port
