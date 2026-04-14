#include "SkiaVkBufferUtils.h"

#include <cstring>
#include <limits>

namespace skia_port {

namespace {

static void SetError(std::string* error, const char* msg) {
    if (error) {
        *error = msg ? msg : "";
    }
}

static bool ValidateImmediateCopyRequest(const VulkanUploadContext& ctx,
                                         VkBuffer src,
                                         VkBuffer dst,
                                         VkDeviceSize size,
                                         std::string* error) {
    if (ctx.device == VK_NULL_HANDLE || ctx.queue == VK_NULL_HANDLE ||
        ctx.commandPool == VK_NULL_HANDLE) {
        SetError(error, "SubmitImmediateCopy: Vulkan upload context is incomplete");
        return false;
    }
    if (src == VK_NULL_HANDLE || dst == VK_NULL_HANDLE || size == 0) {
        SetError(error, "SubmitImmediateCopy: invalid buffer copy request");
        return false;
    }
    return true;
}

static bool AllocatePrimaryCommandBuffer(const VulkanUploadContext& ctx,
                                         VkCommandBuffer* out,
                                         std::string* error) {
    if (!out) {
        SetError(error, "AllocatePrimaryCommandBuffer: out is null");
        return false;
    }
    *out = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = ctx.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(ctx.device, &allocInfo, out) != VK_SUCCESS) {
        SetError(error, "AllocatePrimaryCommandBuffer: vkAllocateCommandBuffers failed");
        return false;
    }
    return true;
}

static void FreeCommandBuffer(const VulkanUploadContext& ctx, VkCommandBuffer* commandBuffer) {
    if (!commandBuffer || *commandBuffer == VK_NULL_HANDLE) {
        return;
    }
    vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, commandBuffer);
    *commandBuffer = VK_NULL_HANDLE;
}

static bool BeginOneTimeCommandBuffer(const VulkanUploadContext& ctx,
                                      VkCommandBuffer commandBuffer,
                                      std::string* error) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        SetError(error, "BeginOneTimeCommandBuffer: vkBeginCommandBuffer failed");
        return false;
    }
    return true;
}

static void RecordBufferCopy(VkCommandBuffer commandBuffer, VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, src, dst, 1, &copyRegion);
}

static bool EndSubmitAndWait(const VulkanUploadContext& ctx,
                             VkCommandBuffer commandBuffer,
                             std::string* error) {
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        SetError(error, "EndSubmitAndWait: vkEndCommandBuffer failed");
        return false;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    if (vkQueueSubmit(ctx.queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        SetError(error, "EndSubmitAndWait: vkQueueSubmit failed");
        return false;
    }
    if (vkQueueWaitIdle(ctx.queue) != VK_SUCCESS) {
        SetError(error, "EndSubmitAndWait: vkQueueWaitIdle failed");
        return false;
    }
    return true;
}

static bool AllocateAndBeginImmediateCommandBuffer(const VulkanUploadContext& ctx,
                                                   VkCommandBuffer* commandBuffer,
                                                   std::string* error) {
    if (!AllocatePrimaryCommandBuffer(ctx, commandBuffer, error)) {
        return false;
    }
    if (!BeginOneTimeCommandBuffer(ctx, *commandBuffer, error)) {
        FreeCommandBuffer(ctx, commandBuffer);
        return false;
    }
    return true;
}

static bool SubmitAndReleaseImmediateCommandBuffer(const VulkanUploadContext& ctx,
                                                   VkCommandBuffer* commandBuffer,
                                                   std::string* error) {
    if (!commandBuffer || *commandBuffer == VK_NULL_HANDLE) {
        SetError(error, "SubmitAndReleaseImmediateCommandBuffer: commandBuffer is null");
        return false;
    }
    if (!EndSubmitAndWait(ctx, *commandBuffer, error)) {
        FreeCommandBuffer(ctx, commandBuffer);
        return false;
    }
    FreeCommandBuffer(ctx, commandBuffer);
    return true;
}

static bool SubmitImmediateCopy(const VulkanUploadContext& ctx,
                                VkBuffer src,
                                VkBuffer dst,
                                VkDeviceSize size,
                                std::string* error) {
    if (!ValidateImmediateCopyRequest(ctx, src, dst, size, error)) {
        return false;
    }

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (!AllocateAndBeginImmediateCommandBuffer(ctx, &commandBuffer, error)) {
        return false;
    }
    RecordBufferCopy(commandBuffer, src, dst, size);
    return SubmitAndReleaseImmediateCommandBuffer(ctx, &commandBuffer, error);
}

static bool CreateHostVisibleBufferWithData(const VulkanUploadContext& ctx,
                                            const void* src,
                                            size_t size,
                                            VkBufferUsageFlags usage,
                                            VkOwnedBuffer* out,
                                            std::string* error) {
    if (!CreateOwnedBuffer(ctx,
                           static_cast<VkDeviceSize>(size),
                           usage,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           out,
                           error)) {
        return false;
    }
    if (!UploadHostVisibleBuffer(ctx, src, size, out, error)) {
        DestroyOwnedBuffer(ctx, out);
        return false;
    }
    return true;
}

static bool ValidateOwnedBufferCreateRequest(const VulkanUploadContext& ctx,
                                             VkDeviceSize size,
                                             VkOwnedBuffer* out,
                                             std::string* error) {
    if (!out) {
        SetError(error, "CreateOwnedBuffer: out is null");
        return false;
    }
    if (ctx.device == VK_NULL_HANDLE || ctx.physicalDevice == VK_NULL_HANDLE) {
        SetError(error, "CreateOwnedBuffer: Vulkan upload context is incomplete");
        return false;
    }
    if (size == 0) {
        SetError(error, "CreateOwnedBuffer: size must be non-zero");
        return false;
    }
    if (size > std::numeric_limits<uint32_t>::max()) {
        SetError(error, "CreateOwnedBuffer: size exceeds VkOwnedBuffer limit");
        return false;
    }
    return true;
}

static bool AllocateAndBindBufferMemory(const VulkanUploadContext& ctx,
                                        VkMemoryPropertyFlags props,
                                        VkOwnedBuffer* out,
                                        std::string* error) {
    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(ctx.device, out->buffer, &memoryRequirements);

    uint32_t memoryTypeIndex = FindMemoryType(ctx,
                                              memoryRequirements.memoryTypeBits,
                                              props,
                                              error);
    if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memoryRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    if (vkAllocateMemory(ctx.device, &allocInfo, nullptr, &out->memory) != VK_SUCCESS) {
        SetError(error, "CreateOwnedBuffer: vkAllocateMemory failed");
        return false;
    }

    if (vkBindBufferMemory(ctx.device, out->buffer, out->memory, 0) != VK_SUCCESS) {
        SetError(error, "CreateOwnedBuffer: vkBindBufferMemory failed");
        return false;
    }

    return true;
}

static void DestroyOwnedBufferPair(const VulkanUploadContext& ctx,
                                   VkOwnedBuffer* first,
                                   VkOwnedBuffer* second) {
    DestroyOwnedBuffer(ctx, first);
    DestroyOwnedBuffer(ctx, second);
}

static bool CreateDeviceLocalDestinationBuffer(const VulkanUploadContext& ctx,
                                               size_t size,
                                               VkBufferUsageFlags usage,
                                               VkOwnedBuffer* out,
                                               std::string* error) {
    return CreateOwnedBuffer(ctx,
                             static_cast<VkDeviceSize>(size),
                             usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             out,
                             error);
}

}  // namespace

uint32_t FindMemoryType(const VulkanUploadContext& ctx,
                        uint32_t typeBits,
                        VkMemoryPropertyFlags props,
                        std::string* error) {
    if (ctx.physicalDevice == VK_NULL_HANDLE) {
        SetError(error, "FindMemoryType: physicalDevice is null");
        return std::numeric_limits<uint32_t>::max();
    }

    VkPhysicalDeviceMemoryProperties memoryProps{};
    vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &memoryProps);
    for (uint32_t i = 0; i < memoryProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (memoryProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }

    SetError(error, "FindMemoryType: no compatible memory type");
    return std::numeric_limits<uint32_t>::max();
}

bool CreateOwnedBuffer(const VulkanUploadContext& ctx,
                       VkDeviceSize size,
                       VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags props,
                       VkOwnedBuffer* out,
                       std::string* error) {
    if (!ValidateOwnedBufferCreateRequest(ctx, size, out, error)) {
        return false;
    }
    DestroyOwnedBuffer(ctx, out);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx.device, &bufferInfo, nullptr, &out->buffer) != VK_SUCCESS) {
        SetError(error, "CreateOwnedBuffer: vkCreateBuffer failed");
        DestroyOwnedBuffer(ctx, out);
        return false;
    }

    if (!AllocateAndBindBufferMemory(ctx, props, out, error)) {
        DestroyOwnedBuffer(ctx, out);
        return false;
    }

    out->sizeBytes = static_cast<uint32_t>(size);
    return true;
}

bool UploadHostVisibleBuffer(const VulkanUploadContext& ctx,
                             const void* src,
                             size_t size,
                             VkOwnedBuffer* dst,
                             std::string* error) {
    if (!dst) {
        SetError(error, "UploadHostVisibleBuffer: dst is null");
        return false;
    }
    if (ctx.device == VK_NULL_HANDLE) {
        SetError(error, "UploadHostVisibleBuffer: device is null");
        return false;
    }
    if (dst->buffer == VK_NULL_HANDLE || dst->memory == VK_NULL_HANDLE) {
        SetError(error, "UploadHostVisibleBuffer: destination buffer is not allocated");
        return false;
    }
    if (size > dst->sizeBytes) {
        SetError(error, "UploadHostVisibleBuffer: source exceeds destination buffer size");
        return false;
    }
    if (size > 0 && !src) {
        SetError(error, "UploadHostVisibleBuffer: src is null");
        return false;
    }

    void* mapped = nullptr;
    if (vkMapMemory(ctx.device, dst->memory, 0, size, 0, &mapped) != VK_SUCCESS) {
        SetError(error, "UploadHostVisibleBuffer: vkMapMemory failed");
        return false;
    }
    if (size > 0) {
        std::memcpy(mapped, src, size);
    }
    vkUnmapMemory(ctx.device, dst->memory);
    return true;
}

bool CreateDeviceLocalBufferWithData(const VulkanUploadContext& ctx,
                                     const void* src,
                                     size_t size,
                                     VkBufferUsageFlags usage,
                                     VkOwnedBuffer* out,
                                     std::string* error) {
    if (!out) {
        SetError(error, "CreateDeviceLocalBufferWithData: out is null");
        return false;
    }
    DestroyOwnedBuffer(ctx, out);

    if (size == 0) {
        return true;
    }
    if (!src) {
        SetError(error, "CreateDeviceLocalBufferWithData: src is null");
        return false;
    }

    VkOwnedBuffer stagingBuffer;
    if (!CreateHostVisibleBufferWithData(ctx,
                                         src,
                                         size,
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                         &stagingBuffer,
                                         error)) {
        return false;
    }

    if (!CreateDeviceLocalDestinationBuffer(ctx, size, usage, out, error)) {
        DestroyOwnedBuffer(ctx, &stagingBuffer);
        return false;
    }

    if (!SubmitImmediateCopy(ctx,
                             stagingBuffer.buffer,
                             out->buffer,
                             static_cast<VkDeviceSize>(size),
                             error)) {
        DestroyOwnedBufferPair(ctx, out, &stagingBuffer);
        return false;
    }

    DestroyOwnedBuffer(ctx, &stagingBuffer);
    return true;
}

void DestroyOwnedBuffer(const VulkanUploadContext& ctx, VkOwnedBuffer* buffer) {
    if (!buffer) {
        return;
    }
    if (ctx.device != VK_NULL_HANDLE && buffer->buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx.device, buffer->buffer, nullptr);
    }
    if (ctx.device != VK_NULL_HANDLE && buffer->memory != VK_NULL_HANDLE) {
        vkFreeMemory(ctx.device, buffer->memory, nullptr);
    }
    buffer->buffer = VK_NULL_HANDLE;
    buffer->memory = VK_NULL_HANDLE;
    buffer->sizeBytes = 0;
}

VkFormat FindDepthStencilFormat(VkPhysicalDevice physicalDevice) {
    if (physicalDevice == VK_NULL_HANDLE) {
        return VK_FORMAT_D24_UNORM_S8_UINT;
    }

    constexpr VkFormat kCandidates[] = {
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D16_UNORM_S8_UINT,
    };
    for (VkFormat format : kCandidates) {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
        if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
            return format;
        }
    }
    return VK_FORMAT_D24_UNORM_S8_UINT;
}

}  // namespace skia_port
