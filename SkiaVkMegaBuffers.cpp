#include "SkiaVkMegaBuffers.h"
#include "SkiaTessPlanVk.h"
#include <cstring>
#include <iostream>

namespace skia_port {

namespace {

void AppendTransferDstBufferBarrier(VkBufferMemoryBarrier* barriers,
                                    uint32_t* barrierCount,
                                    VkBuffer buffer,
                                    VkDeviceSize size,
                                    VkAccessFlags dstAccessMask) {
    if (!barriers || !barrierCount || buffer == VK_NULL_HANDLE || size == 0) {
        return;
    }

    VkBufferMemoryBarrier& barrier = barriers[*barrierCount];
    barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = dstAccessMask;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = size;
    ++*barrierCount;
}

}  // namespace

bool SkiaVkMegaBuffers::init(const VulkanUploadContext& ctx) {
    m_ctx = ctx;
    m_initialized = true;

    m_triangleVertexCapacity = 1024 * 1024;
    if (!resizeBufferPair(m_triangleVertexStagingBuffer, m_triangleVertexStagingMemory, &m_triangleVertexMapped, m_triangleVertexBuffer, m_triangleVertexMemory, 0, m_triangleVertexCapacity, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
        cleanup();
        return false;
    }
    
    m_triangleIndexCapacity = 256 * 1024;
    if (!resizeBufferPair(m_triangleIndexStagingBuffer, m_triangleIndexStagingMemory, &m_triangleIndexMapped, m_triangleIndexBuffer, m_triangleIndexMemory, 0, m_triangleIndexCapacity, VK_BUFFER_USAGE_INDEX_BUFFER_BIT)) {
        cleanup();
        return false;
    }
    
    m_patchCapacity = 1024 * 1024;
    if (!resizeBufferPair(m_patchStagingBuffer, m_patchStagingMemory, &m_patchMapped, m_patchBuffer, m_patchMemory, 0, m_patchCapacity, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
        cleanup();
        return false;
    }
    
    // Instance SSBO at slot 1
    m_instanceCapacity = 16 * 1024; // Objects count
    if (!resizeBufferPair(m_instanceStagingBuffer, m_instanceStagingMemory, &m_instanceMapped, m_instanceSSBO, m_instanceMemory, 0, m_instanceCapacity * sizeof(GPUPathInstance), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
        cleanup();
        return false;
    }
    
    // Command buffers
    m_indirectCmdCapacity = 16 * 1024; // Commands count
    if (!resizeBufferPair(m_indirectCmdStagingBuffer, m_indirectCmdStagingMemory, &m_indirectCmdMapped, m_indirectCmdBuffer, m_indirectCmdMemory, 0, m_indirectCmdCapacity * sizeof(VkDrawIndirectCommand), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)) {
        cleanup();
        return false;
    }
    
    m_indexedIndirectCmdCapacity = 16 * 1024; // Commands count
    if (!resizeBufferPair(m_indexedIndirectCmdStagingBuffer, m_indexedIndirectCmdStagingMemory, &m_indexedIndirectCmdMapped, m_indexedIndirectCmdBuffer, m_indexedIndirectCmdMemory, 0, m_indexedIndirectCmdCapacity * sizeof(VkDrawIndexedIndirectCommand), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)) {
        cleanup();
        return false;
    }

    return true;
}

void SkiaVkMegaBuffers::unmapAll() {
    if (!m_initialized) return;
    if (m_triangleVertexMapped) { vkUnmapMemory(m_ctx.device, m_triangleVertexStagingMemory); m_triangleVertexMapped = nullptr; }
    if (m_triangleIndexMapped) { vkUnmapMemory(m_ctx.device, m_triangleIndexStagingMemory); m_triangleIndexMapped = nullptr; }
    if (m_patchMapped) { vkUnmapMemory(m_ctx.device, m_patchStagingMemory); m_patchMapped = nullptr; }
    if (m_instanceMapped) { vkUnmapMemory(m_ctx.device, m_instanceStagingMemory); m_instanceMapped = nullptr; }
    if (m_indirectCmdMapped) { vkUnmapMemory(m_ctx.device, m_indirectCmdStagingMemory); m_indirectCmdMapped = nullptr; }
    if (m_indexedIndirectCmdMapped) { vkUnmapMemory(m_ctx.device, m_indexedIndirectCmdStagingMemory); m_indexedIndirectCmdMapped = nullptr; }
}

void SkiaVkMegaBuffers::cleanup() {
    if (!m_initialized) return;
    releaseOldBuffers(UINT64_MAX);
    unmapAll();
    auto destroy = [&](VkBuffer& b, VkDeviceMemory& m) {
        if (b != VK_NULL_HANDLE) { vkDestroyBuffer(m_ctx.device, b, nullptr); b = VK_NULL_HANDLE; }
        if (m != VK_NULL_HANDLE) { vkFreeMemory(m_ctx.device, m, nullptr); m = VK_NULL_HANDLE; }
    };
    destroy(m_triangleVertexStagingBuffer, m_triangleVertexStagingMemory);
    destroy(m_triangleIndexStagingBuffer, m_triangleIndexStagingMemory);
    destroy(m_patchStagingBuffer, m_patchStagingMemory);
    destroy(m_instanceStagingBuffer, m_instanceStagingMemory);
    destroy(m_indirectCmdStagingBuffer, m_indirectCmdStagingMemory);
    destroy(m_indexedIndirectCmdStagingBuffer, m_indexedIndirectCmdStagingMemory);

    destroy(m_triangleVertexBuffer, m_triangleVertexMemory);
    destroy(m_triangleIndexBuffer, m_triangleIndexMemory);
    destroy(m_patchBuffer, m_patchMemory);
    destroy(m_instanceSSBO, m_instanceMemory);
    destroy(m_indirectCmdBuffer, m_indirectCmdMemory);
    destroy(m_indexedIndirectCmdBuffer, m_indexedIndirectCmdMemory);

    resetOffsets();
    m_triangleVertexCapacity = 0;
    m_triangleIndexCapacity = 0;
    m_patchCapacity = 0;
    m_instanceCapacity = 0;
    m_indirectCmdCapacity = 0;
    m_indexedIndirectCmdCapacity = 0;
    m_ctx = {};
    m_initialized = false;
}

void SkiaVkMegaBuffers::resetOffsets() {
    m_triangleVertexOffset = 0;
    m_triangleIndexOffset = 0;
    m_patchOffset = 0;
    m_instanceCount = 0;
    m_indirectCmdCount = 0;
    m_indexedIndirectCmdCount = 0;
}

SkiaVkMegaBuffers::OffsetState SkiaVkMegaBuffers::captureOffsetState() const {
    OffsetState state;
    state.triangleVertexOffset = m_triangleVertexOffset;
    state.triangleIndexOffset = m_triangleIndexOffset;
    state.patchOffset = m_patchOffset;
    state.instanceCount = m_instanceCount;
    state.indirectCmdCount = m_indirectCmdCount;
    state.indexedIndirectCmdCount = m_indexedIndirectCmdCount;
    return state;
}

void SkiaVkMegaBuffers::restoreOffsetState(const OffsetState& state) {
    m_triangleVertexOffset = state.triangleVertexOffset;
    m_triangleIndexOffset = state.triangleIndexOffset;
    m_patchOffset = state.patchOffset;
    m_instanceCount = state.instanceCount;
    m_indirectCmdCount = state.indirectCmdCount;
    m_indexedIndirectCmdCount = state.indexedIndirectCmdCount;
}

bool SkiaVkMegaBuffers::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory, void** mappedPtr) {
    if (!m_initialized) return false;
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    if (mappedPtr) {
        *mappedPtr = nullptr;
    }

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(m_ctx.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) return false;

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(m_ctx.device, buffer, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = FindMemoryType(m_ctx, memReqs.memoryTypeBits, properties);

    if (vkAllocateMemory(m_ctx.device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        vkDestroyBuffer(m_ctx.device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }
    if (vkBindBufferMemory(m_ctx.device, buffer, memory, 0) != VK_SUCCESS) {
        vkDestroyBuffer(m_ctx.device, buffer, nullptr);
        vkFreeMemory(m_ctx.device, memory, nullptr);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        return false;
    }

    if (mappedPtr) {
        if (vkMapMemory(m_ctx.device, memory, 0, size, 0, mappedPtr) != VK_SUCCESS) {
            vkDestroyBuffer(m_ctx.device, buffer, nullptr);
            vkFreeMemory(m_ctx.device, memory, nullptr);
            buffer = VK_NULL_HANDLE;
            memory = VK_NULL_HANDLE;
            *mappedPtr = nullptr;
            return false;
        }
    }

    return true;
}

bool SkiaVkMegaBuffers::resizeBufferPair(VkBuffer& stagingBuf, VkDeviceMemory& stagingMem, void** mappedPtr,
                                         VkBuffer& deviceBuf, VkDeviceMemory& deviceMem,
                                         uint32_t currentSize, uint32_t newCapacity, VkBufferUsageFlags usage) {
    if (!m_initialized) return false;
    std::cout << "SkiaVkMegaBuffers: Resizing buffer from " << currentSize << " to " << newCapacity << " bytes." << std::endl;

    VkBuffer newStagingBuf = VK_NULL_HANDLE;
    VkDeviceMemory newStagingMem = VK_NULL_HANDLE;
    void* newMapped = nullptr;
    if (!createBuffer(newCapacity, usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, newStagingBuf, newStagingMem, &newMapped)) {
        return false;
    }

    VkBuffer newDeviceBuf = VK_NULL_HANDLE;
    VkDeviceMemory newDeviceMem = VK_NULL_HANDLE;
    if (!createBuffer(newCapacity, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, newDeviceBuf, newDeviceMem)) {
        if (newMapped) {
            vkUnmapMemory(m_ctx.device, newStagingMem);
        }
        if (newStagingBuf != VK_NULL_HANDLE) {
            vkDestroyBuffer(m_ctx.device, newStagingBuf, nullptr);
        }
        if (newStagingMem != VK_NULL_HANDLE) {
            vkFreeMemory(m_ctx.device, newStagingMem, nullptr);
        }
        return false;
    }

    if (mappedPtr && *mappedPtr && newMapped && currentSize > 0) {
        std::memcpy(newMapped, *mappedPtr, currentSize);
    }

    if (mappedPtr && *mappedPtr) {
        vkUnmapMemory(m_ctx.device, stagingMem);
    }

    if (stagingBuf != VK_NULL_HANDLE) {
        m_deletionQueue.push_back({stagingBuf, stagingMem, m_currentRetireSerial});
    }
    if (deviceBuf != VK_NULL_HANDLE) {
        m_deletionQueue.push_back({deviceBuf, deviceMem, m_currentRetireSerial});
    }

    stagingBuf = newStagingBuf;
    stagingMem = newStagingMem;
    if (mappedPtr) {
        *mappedPtr = newMapped;
    }
    deviceBuf = newDeviceBuf;
    deviceMem = newDeviceMem;
    return true;
}

void SkiaVkMegaBuffers::releaseOldBuffers(uint64_t completedRetireSerial) {
    auto it = m_deletionQueue.begin();
    while (it != m_deletionQueue.end()) {
        if (it->retireSerial <= completedRetireSerial) {
            if (it->buffer != VK_NULL_HANDLE) vkDestroyBuffer(m_ctx.device, it->buffer, nullptr);
            if (it->memory != VK_NULL_HANDLE) vkFreeMemory(m_ctx.device, it->memory, nullptr);
            it = m_deletionQueue.erase(it);
        } else {
            ++it;
        }
    }
}

bool SkiaVkMegaBuffers::flushToDevice(VkCommandBuffer cmd) {
    if (!m_initialized || cmd == VK_NULL_HANDLE) {
        return false;
    }

    auto addCopy = [&](VkBuffer src, VkBuffer dst, uint32_t size) {
        if (size == 0 || src == VK_NULL_HANDLE || dst == VK_NULL_HANDLE) return;
        VkBufferCopy region{};
        region.srcOffset = 0;
        region.dstOffset = 0;
        region.size = size;
        vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    };

    addCopy(m_triangleVertexStagingBuffer, m_triangleVertexBuffer, m_triangleVertexOffset);
    addCopy(m_triangleIndexStagingBuffer, m_triangleIndexBuffer, m_triangleIndexOffset);
    addCopy(m_patchStagingBuffer, m_patchBuffer, m_patchOffset);
    addCopy(m_instanceStagingBuffer, m_instanceSSBO, m_instanceCount * sizeof(GPUPathInstance));
    addCopy(m_indirectCmdStagingBuffer, m_indirectCmdBuffer, m_indirectCmdCount * sizeof(VkDrawIndirectCommand));
    addCopy(m_indexedIndirectCmdStagingBuffer, m_indexedIndirectCmdBuffer, m_indexedIndirectCmdCount * sizeof(VkDrawIndexedIndirectCommand));

    // Keep replay synchronization explicit per resource class so tessellation shaders that read
    // InstanceData are covered, while vertex/index/indirect consumers remain tightly scoped.
    VkBufferMemoryBarrier barriers[6]{};
    uint32_t barrierCount = 0;
    AppendTransferDstBufferBarrier(barriers,
                                   &barrierCount,
                                   m_triangleVertexBuffer,
                                   m_triangleVertexOffset,
                                   VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
    AppendTransferDstBufferBarrier(barriers,
                                   &barrierCount,
                                   m_triangleIndexBuffer,
                                   m_triangleIndexOffset,
                                   VK_ACCESS_INDEX_READ_BIT);
    AppendTransferDstBufferBarrier(barriers,
                                   &barrierCount,
                                   m_patchBuffer,
                                   m_patchOffset,
                                   VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
    AppendTransferDstBufferBarrier(barriers,
                                   &barrierCount,
                                   m_instanceSSBO,
                                   static_cast<VkDeviceSize>(m_instanceCount) *
                                           sizeof(GPUPathInstance),
                                   VK_ACCESS_SHADER_READ_BIT);
    AppendTransferDstBufferBarrier(barriers,
                                   &barrierCount,
                                   m_indirectCmdBuffer,
                                   static_cast<VkDeviceSize>(m_indirectCmdCount) *
                                           sizeof(VkDrawIndirectCommand),
                                   VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    AppendTransferDstBufferBarrier(barriers,
                                   &barrierCount,
                                   m_indexedIndirectCmdBuffer,
                                   static_cast<VkDeviceSize>(m_indexedIndirectCmdCount) *
                                           sizeof(VkDrawIndexedIndirectCommand),
                                   VK_ACCESS_INDIRECT_COMMAND_READ_BIT);

    if (barrierCount > 0) {
        constexpr VkPipelineStageFlags kInstanceShaderStages =
                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
                VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT |
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        constexpr VkPipelineStageFlags kReplayConsumerStages =
                VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                kInstanceShaderStages;

        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             kReplayConsumerStages,
                             0,
                             0,
                             nullptr,
                             barrierCount,
                             barriers,
                             0,
                             nullptr);
    }
    
    m_flushedOffsets = captureOffsetState();
    return true;
}

bool SkiaVkMegaBuffers::appendTriangleData(const void* vertices, uint32_t vertexBytes, uint32_t vStride,
                                           const void* indices, uint32_t indexBytes,
                                           uint32_t* outVertexByteOffset, uint32_t* outIndexByteOffset) {
    uint32_t vertexByteOffset = 0;
    uint32_t indexByteOffset = 0;
    uint32_t vertexOffset = m_triangleVertexOffset;
    uint32_t indexOffset = m_triangleIndexOffset;

    if (vertexBytes > 0) {
        if (!vertices) {
            return false;
        }
        if (vStride > 0) {
            uint32_t remainder = vertexOffset % vStride;
            if (remainder != 0) {
                vertexOffset += (vStride - remainder);
            }
        }
        if (vertexOffset + vertexBytes > m_triangleVertexCapacity) {
            uint32_t newCap = std::max(m_triangleVertexCapacity * 2, vertexOffset + vertexBytes + 1024);
            if (!resizeBufferPair(m_triangleVertexStagingBuffer, m_triangleVertexStagingMemory, &m_triangleVertexMapped, m_triangleVertexBuffer, m_triangleVertexMemory, m_triangleVertexOffset, newCap, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
                return false;
            }
            m_triangleVertexCapacity = newCap;
        }
        if (!m_triangleVertexMapped) {
            return false;
        }
        void* dst = static_cast<uint8_t*>(m_triangleVertexMapped) + vertexOffset;
        std::memcpy(dst, vertices, vertexBytes);
        vertexByteOffset = vertexOffset;
        vertexOffset += vertexBytes;
    }

    if (indexBytes > 0) {
        if (!indices) {
            return false;
        }
        if (indexOffset + indexBytes > m_triangleIndexCapacity) {
            uint32_t newCap = std::max(m_triangleIndexCapacity * 2, indexOffset + indexBytes + 1024);
            if (!resizeBufferPair(m_triangleIndexStagingBuffer, m_triangleIndexStagingMemory, &m_triangleIndexMapped, m_triangleIndexBuffer, m_triangleIndexMemory, m_triangleIndexOffset, newCap, VK_BUFFER_USAGE_INDEX_BUFFER_BIT)) {
                return false;
            }
            m_triangleIndexCapacity = newCap;
        }
        if (!m_triangleIndexMapped) {
            return false;
        }
        void* dst = static_cast<uint8_t*>(m_triangleIndexMapped) + indexOffset;
        std::memcpy(dst, indices, indexBytes);
        indexByteOffset = indexOffset;
        indexOffset += indexBytes;
    }

    m_triangleVertexOffset = vertexOffset;
    m_triangleIndexOffset = indexOffset;
    if (outVertexByteOffset) {
        *outVertexByteOffset = vertexByteOffset;
    }
    if (outIndexByteOffset) {
        *outIndexByteOffset = indexByteOffset;
    }
    return true;
}

bool SkiaVkMegaBuffers::appendPatchInstanceData(const void* data, uint32_t bytes,
                                                uint32_t stride, uint32_t* outByteOffset) {
    uint32_t patchByteOffset = 0;
    uint32_t patchOffset = m_patchOffset;

    if (bytes > 0) {
        if (!data) {
            return false;
        }
        if (stride > 0) {
            uint32_t remainder = patchOffset % stride;
            if (remainder != 0) {
                patchOffset += (stride - remainder);
            }
        }
        if (patchOffset + bytes > m_patchCapacity) {
            uint32_t newCap = std::max(m_patchCapacity * 2, patchOffset + bytes + 1024 * 64);
            if (!resizeBufferPair(m_patchStagingBuffer, m_patchStagingMemory, &m_patchMapped, m_patchBuffer, m_patchMemory, m_patchOffset, newCap, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
                return false;
            }
            m_patchCapacity = newCap;
        }
        if (!m_patchMapped) {
            return false;
        }
        void* dst = static_cast<uint8_t*>(m_patchMapped) + patchOffset;
        std::memcpy(dst, data, bytes);
        patchByteOffset = patchOffset;
        patchOffset += bytes;
    }

    m_patchOffset = patchOffset;
    if (outByteOffset) {
        *outByteOffset = patchByteOffset;
    }
    return true;
}

bool SkiaVkMegaBuffers::appendTessPassUpload(const VkTessPassUploadBytes& upload,
                                             VkTessPassUploadOffsets* outOffsets) {
    if (outOffsets) {
        *outOffsets = {};
    }

    const OffsetState stateBeforeAppend = captureOffsetState();
    uint32_t vertexByteOffset = 0;
    uint32_t indexByteOffset = 0;
    const bool hasVertexBytes = !upload.vertexBytes.empty();
    const bool hasIndexBytes = !upload.indexBytes.empty();
    if (hasVertexBytes || hasIndexBytes) {
        if (!appendTriangleData(hasVertexBytes ? upload.vertexBytes.data() : nullptr,
                                static_cast<uint32_t>(upload.vertexBytes.size()),
                                upload.vertexStrideBytes,
                                hasIndexBytes ? upload.indexBytes.data() : nullptr,
                                static_cast<uint32_t>(upload.indexBytes.size()),
                                &vertexByteOffset,
                                &indexByteOffset)) {
            restoreOffsetState(stateBeforeAppend);
            return false;
        }
    }

    uint32_t instanceByteOffset = 0;
    if (!upload.instanceBytes.empty()) {
        if (!appendPatchInstanceData(upload.instanceBytes.data(),
                                     static_cast<uint32_t>(upload.instanceBytes.size()),
                                     upload.instanceStrideBytes,
                                     &instanceByteOffset)) {
            restoreOffsetState(stateBeforeAppend);
            return false;
        }
    }

    if (outOffsets) {
        outOffsets->vertexByteOffset = vertexByteOffset;
        outOffsets->indexByteOffset = indexByteOffset;
        outOffsets->instanceByteOffset = instanceByteOffset;
    }
    return true;
}

bool SkiaVkMegaBuffers::appendGlobalInstance(const GPUPathInstance& inst, uint32_t* outIndex) {
    const uint32_t instanceIndex = m_instanceCount;
    if (instanceIndex + 1 > m_instanceCapacity) {
        uint32_t newCap = std::max(m_instanceCapacity * 2, instanceIndex + 1024);
        if (!resizeBufferPair(m_instanceStagingBuffer, m_instanceStagingMemory, &m_instanceMapped, m_instanceSSBO, m_instanceMemory, m_instanceCount * sizeof(GPUPathInstance), newCap * sizeof(GPUPathInstance), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
            return false;
        }
        m_instanceCapacity = newCap;
    }
    if (!m_instanceMapped) {
        return false;
    }
    GPUPathInstance* dst = static_cast<GPUPathInstance*>(m_instanceMapped) + instanceIndex;
    *dst = inst;
    if (outIndex) {
        *outIndex = instanceIndex;
    }
    m_instanceCount = instanceIndex + 1;
    return true;
}

bool SkiaVkMegaBuffers::appendDrawIndirectCmd(const VkDrawIndirectCommand& cmd) {
    const uint32_t cmdIndex = m_indirectCmdCount;
    if (cmdIndex + 1 > m_indirectCmdCapacity) {
        uint32_t newCap = std::max(m_indirectCmdCapacity * 2, cmdIndex + 256);
        if (!resizeBufferPair(m_indirectCmdStagingBuffer, m_indirectCmdStagingMemory, &m_indirectCmdMapped, m_indirectCmdBuffer, m_indirectCmdMemory, m_indirectCmdCount * sizeof(VkDrawIndirectCommand), newCap * sizeof(VkDrawIndirectCommand), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)) {
            return false;
        }
        m_indirectCmdCapacity = newCap;
    }
    if (!m_indirectCmdMapped) {
        return false;
    }
    VkDrawIndirectCommand* dst = static_cast<VkDrawIndirectCommand*>(m_indirectCmdMapped) + cmdIndex;
    *dst = cmd;
    m_indirectCmdCount = cmdIndex + 1;
    return true;
}

bool SkiaVkMegaBuffers::appendDrawIndexedIndirectCmd(const VkDrawIndexedIndirectCommand& cmd) {
    const uint32_t cmdIndex = m_indexedIndirectCmdCount;
    if (cmdIndex + 1 > m_indexedIndirectCmdCapacity) {
        uint32_t newCap = std::max(m_indexedIndirectCmdCapacity * 2, cmdIndex + 256);
        if (!resizeBufferPair(m_indexedIndirectCmdStagingBuffer, m_indexedIndirectCmdStagingMemory, &m_indexedIndirectCmdMapped, m_indexedIndirectCmdBuffer, m_indexedIndirectCmdMemory, m_indexedIndirectCmdCount * sizeof(VkDrawIndexedIndirectCommand), newCap * sizeof(VkDrawIndexedIndirectCommand), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)) {
            return false;
        }
        m_indexedIndirectCmdCapacity = newCap;
    }
    if (!m_indexedIndirectCmdMapped) {
        return false;
    }
    VkDrawIndexedIndirectCommand* dst = static_cast<VkDrawIndexedIndirectCommand*>(m_indexedIndirectCmdMapped) + cmdIndex;
    *dst = cmd;
    m_indexedIndirectCmdCount = cmdIndex + 1;
    return true;
}

} // namespace skia_port
