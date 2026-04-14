#pragma once

#include "PathMeshRuntimeTypes.h"
#include "SkiaPathMeshPort.h"
#include "SkiaProbeTypes.h"
#include "SkiaSvgPathLoader.h"
#include "SkiaVkMegaBuffers.h"

#include <memory>
#include <string>
#include <vector>

namespace skia_port {

class PathMeshWindowPipelineBundle;

class PathMeshCopiedRuntime {
public:
    PathMeshCopiedRuntime() = default;
    ~PathMeshCopiedRuntime() = default;

    void cleanup(const PathMeshRuntimeContext& ctx);

    bool captureTessPlans(VkSampleCountFlagBits msaaSamples,
                          const std::vector<SvgFillShape>& fillShapes,
                          const std::vector<SvgStrokePath>& strokePaths,
                          const PatchPrepareOptions& prepareOptions,
                          std::string* error = nullptr);

    bool uploadTessPlans(const PathMeshRuntimeContext& ctx,
                         PathMeshWindowPipelineBundle* pipelines,
                         bool probeModeEnabled,
                         std::string* error = nullptr,
                         PathMeshUploadStats* stats = nullptr);

    void resetUploadedTessGpuState(const PathMeshRuntimeContext& ctx);
    void destroyTessPlans(const PathMeshRuntimeContext& ctx);
    bool waitForUploadIdle(const PathMeshRuntimeContext& ctx, std::string* error = nullptr) const;

    bool clearProbeRecords(const PathMeshRuntimeContext& ctx, std::string* error = nullptr);
    bool readProbeRecords(const PathMeshRuntimeContext& ctx,
                          std::vector<GpuProbeRecord>* outRecords,
                          std::string* error = nullptr);

    void recordDrawCommands(VkCommandBuffer cmd,
                            PathMeshWindowPipelineBundle* pipelines,
                            VkExtent2D swapchainExtent,
                            bool probeModeEnabled);

    bool hasTessPlans() const { return !m_tessPlans.empty(); }
    bool hasReplayDrawData() const;
    bool hasProbeResources() const {
        return m_probeMemory != VK_NULL_HANDLE && m_probeDescriptorSet != VK_NULL_HANDLE;
    }

    size_t tessPlanCount() const { return m_tessPlans.size(); }
    size_t instanceCount() const { return m_instances.size(); }
    uint32_t indirectCmdCount() const { return m_megaBuffers.getIndirectCmdCount(); }
    uint32_t indexedIndirectCmdCount() const {
        return m_megaBuffers.getIndexedIndirectCmdCount();
    }
    size_t probeRecordCount() const { return m_probeRecordCount; }
    const TessExecutorPrepareStats& replayStats() const { return m_lastPrepareStats; }
    double lastReplayRecordMs() const { return m_lastReplayRecordMs; }

    const std::vector<std::vector<TessProbePassInfo>>& probePassInfos() const {
        return m_probePassInfos;
    }

private:
    VulkanUploadContext makeUploadContext(const PathMeshRuntimeContext& ctx) const;
    uint32_t findMemoryType(const PathMeshRuntimeContext& ctx,
                            uint32_t typeBits,
                            VkMemoryPropertyFlags props) const;
    void createBuffer(const PathMeshRuntimeContext& ctx,
                      VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props,
                      VkBuffer& outBuffer,
                      VkDeviceMemory& outMemory) const;
    void uploadBuffer(VkDevice device, VkDeviceMemory memory, const void* src, size_t size) const;
    void destroyBuffer(VkDevice device, VkBuffer& buffer, VkDeviceMemory& memory) const;
    VkShaderModule createShaderModule(VkDevice device, const uint32_t* data, size_t sizeBytes) const;

    bool rebuildProbeResources(const PathMeshRuntimeContext& ctx,
                               PathMeshWindowPipelineBundle* pipelines,
                               std::string* error = nullptr);
    void destroyProbeResources(const PathMeshRuntimeContext& ctx);

    std::vector<TessCapturePlan> m_tessPlans;
    SkiaVkMegaBuffers m_megaBuffers;
    std::vector<GPUPathInstance> m_instances;
    std::vector<std::vector<TessProbePassInfo>> m_probePassInfos;
    size_t m_probeRecordCount = 0;
    VkBuffer m_probeBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_probeMemory = VK_NULL_HANDLE;
    VkDescriptorPool m_probeDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_probeDescriptorSet = VK_NULL_HANDLE;
    VkFence m_uploadFence = VK_NULL_HANDLE;
    uint64_t m_uploadRetireSerial = 0;
    uint64_t m_completedUploadRetireSerial = 0;
    TessExecutorPrepareStats m_lastPrepareStats{};
    double m_lastReplayRecordMs = 0.0;
};

}  // namespace skia_port
