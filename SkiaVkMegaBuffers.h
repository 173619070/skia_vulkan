#pragma once
#include <vulkan/vulkan.h>
#include "SkiaVkUploadContext.h"
#include "SkiaVkBufferUtils.h"
#include <vector>

namespace skia_port {

struct VkTessPassUploadBytes;
struct VkTessPassUploadOffsets;

struct GPUPathInstance {
    float shaderMatrixRow0[4];
    float shaderMatrixRow1[4];
    float fillColor[4];
    float strokeTessArgs[4];
};

class SkiaVkMegaBuffers {
public:
    SkiaVkMegaBuffers() = default;
    ~SkiaVkMegaBuffers() = default;

    bool init(const VulkanUploadContext& ctx);
    bool isInitialized() const { return m_initialized; }
    void cleanup();

    void setRetireSerial(uint64_t retireSerial) { m_currentRetireSerial = retireSerial; }
    void releaseOldBuffers(uint64_t completedRetireSerial);
    bool flushToDevice(VkCommandBuffer cmd);

    void resetOffsets();
    void unmapAll();

    bool appendTriangleData(const void* vertices, uint32_t vertexBytes, uint32_t vStride,
                            const void* indices, uint32_t indexBytes,
                            uint32_t* outVertexByteOffset, uint32_t* outIndexByteOffset);

    bool appendPatchInstanceData(const void* data, uint32_t bytes,
                                 uint32_t stride, uint32_t* outByteOffset);
    bool appendTessPassUpload(const VkTessPassUploadBytes& upload,
                              VkTessPassUploadOffsets* outOffsets);

    bool appendGlobalInstance(const GPUPathInstance& inst, uint32_t* outIndex);

    bool appendDrawIndirectCmd(const VkDrawIndirectCommand& cmd);
    bool appendDrawIndexedIndirectCmd(const VkDrawIndexedIndirectCommand& cmd);

    // Helpers
    VkBuffer getTriangleVertexBuffer() const { return m_triangleVertexBuffer; }
    VkBuffer getTriangleIndexBuffer() const { return m_triangleIndexBuffer; }
    VkBuffer getPatchBuffer() const { return m_patchBuffer; }
    VkBuffer getInstanceSSBO() const { return m_instanceSSBO; }
    VkBuffer getIndirectCmdBuffer() const { return m_indirectCmdBuffer; }
    VkBuffer getIndexedIndirectCmdBuffer() const { return m_indexedIndirectCmdBuffer; }

    uint32_t getIndirectCmdCount() const { return m_indirectCmdCount; }
    uint32_t getIndexedIndirectCmdCount() const { return m_indexedIndirectCmdCount; }

private:
    struct OffsetState {
        uint32_t triangleVertexOffset = 0;
        uint32_t triangleIndexOffset = 0;
        uint32_t patchOffset = 0;
        uint32_t instanceCount = 0;
        uint32_t indirectCmdCount = 0;
        uint32_t indexedIndirectCmdCount = 0;
    };

    OffsetState captureOffsetState() const;
    void restoreOffsetState(const OffsetState& state);

    struct GarbageItem {
        VkBuffer buffer;
        VkDeviceMemory memory;
        uint64_t retireSerial;
    };
    std::vector<GarbageItem> m_deletionQueue;
    uint64_t m_currentRetireSerial = 0;
    
    // Track how many bytes have been written so flushToDevice only copies what's needed
    OffsetState m_flushedOffsets;

    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory, void** mappedPtr = nullptr);
    bool resizeBufferPair(VkBuffer& stagingBuf, VkDeviceMemory& stagingMem, void** mappedPtr,
                          VkBuffer& deviceBuf, VkDeviceMemory& deviceMem,
                          uint32_t currentSize, uint32_t newCapacity, VkBufferUsageFlags usage);

    VulkanUploadContext m_ctx;
    bool m_initialized = false;

    VkBuffer m_triangleVertexStagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_triangleVertexStagingMemory = VK_NULL_HANDLE;
    VkBuffer m_triangleVertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_triangleVertexMemory = VK_NULL_HANDLE;
    void* m_triangleVertexMapped = nullptr;
    uint32_t m_triangleVertexOffset = 0;
    uint32_t m_triangleVertexCapacity = 0;

    VkBuffer m_triangleIndexStagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_triangleIndexStagingMemory = VK_NULL_HANDLE;
    VkBuffer m_triangleIndexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_triangleIndexMemory = VK_NULL_HANDLE;
    void* m_triangleIndexMapped = nullptr;
    uint32_t m_triangleIndexOffset = 0;
    uint32_t m_triangleIndexCapacity = 0;

    VkBuffer m_patchStagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_patchStagingMemory = VK_NULL_HANDLE;
    VkBuffer m_patchBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_patchMemory = VK_NULL_HANDLE;
    void* m_patchMapped = nullptr;
    uint32_t m_patchOffset = 0;
    uint32_t m_patchCapacity = 0;

    VkBuffer m_instanceStagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_instanceStagingMemory = VK_NULL_HANDLE;
    VkBuffer m_instanceSSBO = VK_NULL_HANDLE;
    VkDeviceMemory m_instanceMemory = VK_NULL_HANDLE;
    void* m_instanceMapped = nullptr;
    uint32_t m_instanceCount = 0;
    uint32_t m_instanceCapacity = 0;

    VkBuffer m_indirectCmdStagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_indirectCmdStagingMemory = VK_NULL_HANDLE;
    VkBuffer m_indirectCmdBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_indirectCmdMemory = VK_NULL_HANDLE;
    void* m_indirectCmdMapped = nullptr;
    uint32_t m_indirectCmdCount = 0;
    uint32_t m_indirectCmdCapacity = 0;

    VkBuffer m_indexedIndirectCmdStagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_indexedIndirectCmdStagingMemory = VK_NULL_HANDLE;
    VkBuffer m_indexedIndirectCmdBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_indexedIndirectCmdMemory = VK_NULL_HANDLE;
    void* m_indexedIndirectCmdMapped = nullptr;
    uint32_t m_indexedIndirectCmdCount = 0;
    uint32_t m_indexedIndirectCmdCapacity = 0;
};

} // namespace skia_port
