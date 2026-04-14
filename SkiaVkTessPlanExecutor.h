#pragma once

#include "SkiaProbeTypes.h"
#include "SkiaTessPlanVk.h"
#include "SkiaVkMegaBuffers.h"
#include <vulkan/vulkan.h>
#include <array>
#include <cstdint>
#include <vector>

namespace skia_port {

struct ExecutorContext {
    VkDevice device = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    uint32_t subpass = 0;
    VkExtent2D msaaExtent{};
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
};

struct TessPushConstants {
    float viewportScaleX = 1.0f;
    float viewportScaleY = 1.0f;
    float maxResolveLevel = 5.0f;
    uint32_t baseCmdIndex = 0;
    // The instance specific data (fillColor, shaderMatrix, etc) are moved to the SSBO.
    // Probe fields remain for debugging logic.
    uint32_t probeShapeIndex = 0;
    uint32_t probePassIndex = 0;
    uint32_t probeVertexBase = 0;
    uint32_t probeVertexCount = 0;
    uint32_t probeInstanceBase = 0;
    uint32_t probeInstanceCount = 0;
    uint32_t probeRecordBase = 0;
    uint32_t probeFlags = 0;
};

struct ExecutorShaderModules {
    VkShaderModule patchVert = VK_NULL_HANDLE;
    VkShaderModule patchProbeVert = VK_NULL_HANDLE;
    VkShaderModule patchFanVert = VK_NULL_HANDLE;
    VkShaderModule patchFanProbeVert = VK_NULL_HANDLE;
    VkShaderModule patchTesc = VK_NULL_HANDLE;
    VkShaderModule patchTese = VK_NULL_HANDLE;
    VkShaderModule patchFrag = VK_NULL_HANDLE;
    VkShaderModule strokeVert = VK_NULL_HANDLE;
    VkShaderModule strokeProbeVert = VK_NULL_HANDLE;
    VkShaderModule strokeFrag = VK_NULL_HANDLE;
    VkShaderModule bboxVert = VK_NULL_HANDLE;
    VkShaderModule bboxProbeVert = VK_NULL_HANDLE;
    VkShaderModule hullVert = VK_NULL_HANDLE;
    VkShaderModule hullProbeVert = VK_NULL_HANDLE;
    VkShaderModule tessFillVert = VK_NULL_HANDLE;
    VkShaderModule tessFillProbeVert = VK_NULL_HANDLE;
    VkShaderModule tessFillFrag = VK_NULL_HANDLE;
    bool useProbeShaderVariants = false;
};

constexpr size_t kTessPlanPassKindCount =
        static_cast<size_t>(TessPlanPassKind::kStrokePatches) + 1;

struct TessExecutorPrepareStats {
    uint32_t planCount = 0;
    uint32_t passCount = 0;
    uint32_t drawCmdCount = 0;
    uint32_t batchCount = 0;
    uint32_t mergedDrawCmdCount = 0;
    uint32_t indirectCmdCount = 0;
    uint32_t indexedIndirectCmdCount = 0;
    uint32_t globalInstanceCount = 0;
    uint64_t vertexUploadBytes = 0;
    uint64_t indexUploadBytes = 0;
    uint64_t instanceUploadBytes = 0;
    std::array<uint32_t, kTessPlanPassKindCount> passCountByKind{};
    std::array<uint32_t, kTessPlanPassKindCount> drawCmdCountByKind{};
    std::array<uint32_t, kTessPlanPassKindCount> batchCountByKind{};
};

struct ExecutorReplayDrawRef {
    uint32_t shapeIndex = 0;
    uint32_t passIndex = 0;
    uint32_t drawIndex = 0;
    TessPlanPassKind passKind = TessPlanPassKind::kUnknown;
};

struct ExecutorCachedReplayOrders {
    std::vector<ExecutorReplayDrawRef> originalDrawRefs;
    std::vector<ExecutorReplayDrawRef> windowedDrawRefs;
};

struct ExecutorReplayPassDescriptor {
    TessProgramInfo programInfo{};
    GPUPathInstance instanceTemplate{};
    TessPlanPassKind passKind = TessPlanPassKind::kUnknown;
    uint32_t probeFlagsTemplate = 0;
    bool hasCustomMaxResolveLevel = false;
    float maxResolveLevel = 0.0f;
};

struct ExecutorReplayDrawDescriptor {
    bool isIndexed = false;
    bool usesPatchBaseForFirstInstance = false;
    uint32_t elementCount = 0;
    uint32_t instanceCount = 0;
    uint32_t firstIndexOffset = 0;
    int32_t vertexOffset = 0;
    uint32_t firstVertexOffset = 0;
    uint32_t firstInstanceOffset = 0;
    uint32_t globalInstanceOrdinal = 0;
    uint32_t commandStreamIndex = 0;
};

struct ExecutorReplayBatchSeed {
    uint32_t firstDrawRefIndex = 0;
    uint32_t drawCount = 0;
};

struct ExecutorCachedReplayBatchPlans {
    std::vector<ExecutorReplayBatchSeed> originalBatches;
    std::vector<ExecutorReplayBatchSeed> windowedBatches;
};

struct ExecutorCachedReplayPrepareStats {
    uint32_t planCount = 0;
    uint32_t passCount = 0;
    uint32_t drawCmdCount = 0;
    uint32_t indirectCmdCount = 0;
    uint32_t indexedIndirectCmdCount = 0;
    uint32_t globalInstanceCount = 0;
    uint64_t vertexUploadBytes = 0;
    uint64_t indexUploadBytes = 0;
    uint64_t instanceUploadBytes = 0;
    std::array<uint32_t, kTessPlanPassKindCount> passCountByKind{};
    std::array<uint32_t, kTessPlanPassKindCount> drawCmdCountByKind{};
};

class SkiaVkTessPlanExecutor {
public:
    SkiaVkTessPlanExecutor();
    ~SkiaVkTessPlanExecutor();

    bool init(const ExecutorContext& ctx);
    void cleanup();

    VkDescriptorSetLayout getDescriptorSetLayout() const { return m_instanceDescriptorSetLayout; }
    VkDescriptorSetLayout getInstanceDescriptorSetLayout() const {
        return m_instanceDescriptorSetLayout;
    }
    VkDescriptorSetLayout getProbeDescriptorSetLayout() const {
        return m_probeDescriptorSetLayout;
    }
    const TessExecutorPrepareStats& getPrepareStats() const { return m_prepareStats; }

    bool preparePipelines(const VulkanUploadContext& uploadCtx,
                          const std::vector<TessCapturePlan>& plans,
                          const std::vector<GPUPathInstance>& instances,
                          SkiaVkMegaBuffers& replayBuffers,
                          const std::vector<std::vector<VkTessPassUploadOffsets>>*
                                  cachedGeometryUploadOffsets,
                          VkPipelineLayout pipelineLayout,
                          const ExecutorShaderModules& shaders,
                          const std::vector<std::vector<VkTessPassUploadBytes>>* cachedPassUploads =
                                  nullptr,
                          const ExecutorCachedReplayOrders* cachedReplayOrders = nullptr,
                          const ExecutorCachedReplayPrepareStats* cachedReplayPrepareStats =
                                  nullptr,
                          const ExecutorCachedReplayBatchPlans* cachedReplayBatchPlans =
                                  nullptr,
                          const std::vector<std::vector<ExecutorReplayPassDescriptor>>*
                                  cachedReplayPassDescriptors =
                                  nullptr,
                          const std::vector<std::vector<std::vector<ExecutorReplayDrawDescriptor>>>*
                                  cachedReplayDrawDescriptors =
                                  nullptr,
                          bool reuseExistingUpload = false);

    struct MdiBatch {
        VkPipeline pipeline = VK_NULL_HANDLE;
        TessPlanPassKind passKind = TessPlanPassKind::kUnknown;
        uint32_t firstCmdIndex = 0;
        uint32_t cmdCount = 0;
        bool isIndexed = false;
        bool usesReplayPatchBuffer = false;
        uint32_t baseSSBOIndex = 0;
        uint32_t shapeIndex = 0;
        uint32_t passIndex = 0;
        uint32_t probeFlagsTemplate = 0;
        bool hasCustomMaxResolveLevel = false;
        float maxResolveLevel = 0.0f;
    };

    const std::vector<MdiBatch>& getBatches() const { return m_batches; }
    void resetPreparedBatches();
    void bindSSBO(const VulkanUploadContext& uploadCtx, SkiaVkMegaBuffers& replayBuffers);
    void execute(VkCommandBuffer cmd,
                 SkiaVkMegaBuffers& geometryBuffers,
                 SkiaVkMegaBuffers& replayBuffers,
                 VkPipelineLayout pipelineLayout,
                 const TessPushConstants& pc,
                 VkDescriptorSet probeDescriptorSet = VK_NULL_HANDLE,
                 const std::vector<std::vector<TessProbePassInfo>>* probePasses = nullptr);

    void executeBatches(VkCommandBuffer cmd,
                        SkiaVkMegaBuffers& geometryBuffers,
                        SkiaVkMegaBuffers& replayBuffers,
                        VkPipelineLayout pipelineLayout,
                        const TessPushConstants& pc,
                        uint32_t firstBatchIndex,
                        uint32_t batchCount,
                        VkPipeline& inOutPipeline,
                        bool& inOutHasBoundVertexBuffers,
                        bool& inOutUsesReplayPatchBuffer,
                        VkDescriptorSet probeDescriptorSet = VK_NULL_HANDLE,
                        const std::vector<std::vector<TessProbePassInfo>>* probePasses = nullptr);

private:
    VkPipeline getOrCreatePipeline(const TessProgramInfo& info,
                                   VkPipelineLayout pipelineLayout,
                                   const ExecutorShaderModules& shaders);

    ExecutorContext m_ctx;

    struct CachedPipeline {
        TessProgramInfo info;
        bool useProbeShaderVariants = false;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };
    std::vector<CachedPipeline> m_pipelines;

    VkDescriptorSetLayout m_instanceDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_probeDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_instanceDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_instanceDescriptorSet = VK_NULL_HANDLE;
    
    // Batches are cached by preparePipelines() for execute()/executeBatches()
    std::vector<MdiBatch> m_batches;
    TessExecutorPrepareStats m_prepareStats;
};

} // namespace skia_port
