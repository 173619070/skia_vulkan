#include "SkiaPathMeshRuntime.h"

#include "PathMeshWindowPipelineBundle.h"
#include "SkiaDemoFailPoint.h"
#include "SkiaProbeLayout.h"
#include "generated_shaders/bbox_probe_vert_spv.h"
#include "generated_shaders/bbox_vert_spv.h"
#include "generated_shaders/hull_probe_vert_spv.h"
#include "generated_shaders/hull_vert_spv.h"
#include "generated_shaders/patch_fan_probe_vert_spv.h"
#include "generated_shaders/patch_fan_vert_spv.h"
#include "generated_shaders/patch_frag_spv.h"
#include "generated_shaders/patch_probe_vert_spv.h"
#include "generated_shaders/patch_tesc_spv.h"
#include "generated_shaders/patch_tese_spv.h"
#include "generated_shaders/patch_vert_spv.h"
#include "generated_shaders/stroke_frag_spv.h"
#include "generated_shaders/stroke_probe_vert_spv.h"
#include "generated_shaders/stroke_vert_spv.h"
#include "generated_shaders/tess_fill_frag_spv.h"
#include "generated_shaders/tess_fill_probe_vert_spv.h"
#include "generated_shaders/tess_fill_vert_spv.h"

#include <ported_skia/src/gpu/tessellate/Tessellation.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

void SetError(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

std::string MakeInjectedFailMessage(const char* where) {
    std::string message = "Injected failpoint";
    if (where && *where) {
        message += ": ";
        message += where;
    }
    return message;
}

double PerfNowMs() {
    using PerfClock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(PerfClock::now().time_since_epoch())
            .count();
}

double PerfElapsedMs(double startMs) {
    return PerfNowMs() - startMs;
}

struct ScopedShaderModules {
    VkDevice device = VK_NULL_HANDLE;
    std::vector<VkShaderModule> modules;

    explicit ScopedShaderModules(VkDevice inDevice) : device(inDevice) {}
    ScopedShaderModules(const ScopedShaderModules&) = delete;
    ScopedShaderModules& operator=(const ScopedShaderModules&) = delete;

    ~ScopedShaderModules() {
        if (device == VK_NULL_HANDLE) {
            return;
        }
        for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
            if (*it != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, *it, nullptr);
            }
        }
    }

    VkShaderModule add(VkShaderModule module) {
        if (module != VK_NULL_HANDLE) {
            modules.push_back(module);
        }
        return module;
    }
};

}  // namespace

namespace skia_port {

void PathMeshCopiedRuntime::cleanup(const PathMeshRuntimeContext& ctx) {
    destroyTessPlans(ctx);
    m_megaBuffers.cleanup();
    if (ctx.device != VK_NULL_HANDLE && m_uploadFence != VK_NULL_HANDLE) {
        vkDestroyFence(ctx.device, m_uploadFence, nullptr);
    }
    m_uploadFence = VK_NULL_HANDLE;
    m_uploadRetireSerial = 0;
    m_completedUploadRetireSerial = 0;
    m_lastPrepareStats = {};
}

bool PathMeshCopiedRuntime::captureTessPlans(VkSampleCountFlagBits msaaSamples,
                                             const std::vector<SvgFillShape>& fillShapes,
                                             const std::vector<SvgStrokePath>& strokePaths,
                                             const PatchPrepareOptions& prepareOptions,
                                             std::string* error) {
    m_tessPlans.clear();

    const AAMode aaType = (msaaSamples != VK_SAMPLE_COUNT_1_BIT) ? AAMode::kMSAA : AAMode::kNone;

    auto captureShape = [&](const SkPath& path,
                            const float color[4],
                            const PathDrawOptions& drawOpts) -> bool {
        TessCapturePlan plan;
        std::string localError;
        if (!CapturePathDrawPlanOriginalSkia(path, drawOpts, &plan, &localError)) {
            m_tessPlans.clear();
            SetError(error, "CapturePathDrawPlanOriginalSkia failed: " + localError);
            return false;
        }

        plan.color[0] = color[0];
        plan.color[1] = color[1];
        plan.color[2] = color[2];
        plan.color[3] = color[3];
        m_tessPlans.push_back(std::move(plan));
        return true;
    };

    for (const auto& shape : fillShapes) {
        PathDrawOptions drawOpts;
        drawOpts.isStroke = false;
        drawOpts.aaType = aaType;
        drawOpts.patchOptions = prepareOptions;
        const float color[4] = {shape.r, shape.g, shape.b, shape.a};
        if (!captureShape(shape.skPath, color, drawOpts)) {
            return false;
        }
    }

    for (const auto& stroke : strokePaths) {
        PathDrawOptions drawOpts;
        drawOpts.isStroke = true;
        drawOpts.aaType = aaType;
        drawOpts.patchOptions = prepareOptions;

        StrokeOptions strokeOptions;
        strokeOptions.width = stroke.width;
        strokeOptions.miterLimit = stroke.miterLimit;
        switch (stroke.lineCap) {
            case SvgLineCap::kRound:
                strokeOptions.cap = StrokeCap::kRound;
                break;
            case SvgLineCap::kSquare:
                strokeOptions.cap = StrokeCap::kSquare;
                break;
            default:
                strokeOptions.cap = StrokeCap::kButt;
                break;
        }
        switch (stroke.lineJoin) {
            case SvgLineJoin::kRound:
                strokeOptions.join = StrokeJoin::kRound;
                break;
            case SvgLineJoin::kBevel:
                strokeOptions.join = StrokeJoin::kBevel;
                break;
            default:
                strokeOptions.join = StrokeJoin::kMiter;
                break;
        }

        drawOpts.strokeOptions = strokeOptions;
        const float color[4] = {stroke.r, stroke.g, stroke.b, stroke.a};
        if (!captureShape(stroke.skPath, color, drawOpts)) {
            return false;
        }
    }

    return true;
}

bool PathMeshCopiedRuntime::uploadTessPlans(const PathMeshRuntimeContext& ctx,
                                            PathMeshWindowPipelineBundle* pipelines,
                                            bool probeModeEnabled,
                                            std::string* error,
                                            PathMeshUploadStats* stats) {
    PathMeshUploadStats localStats;
    PathMeshUploadStats* uploadStats = stats ? stats : &localStats;
    *uploadStats = {};

    auto failUpload = [&](const std::string& message) {
        if (pipelines) {
            uploadStats->executorPrepareStats = pipelines->replayStats();
        }
        resetUploadedTessGpuState(ctx);
        SetError(error, message);
        return false;
    };

    if (pipelines &&
        demo_failpoint::ConsumeDemoFailPoint(demo_failpoint::DemoFailPoint::kExecutorInit)) {
        return failUpload(MakeInjectedFailMessage("PathMeshCopiedRuntime/uploadTessPlans/executor"));
    }

    if (pipelines &&
        !pipelines->EnsureExecutorInitialized(
                ctx.device, ctx.renderPass, ctx.swapchainExtent, ctx.msaaSamples, error)) {
        resetUploadedTessGpuState(ctx);
        return false;
    }

    const double setupStartMs = PerfNowMs();
    resetUploadedTessGpuState(ctx);
    if (m_tessPlans.empty()) {
        m_lastPrepareStats = {};
        uploadStats->setupInstancesMs = PerfElapsedMs(setupStartMs);
        return true;
    }

    const VulkanUploadContext uploadCtx = makeUploadContext(ctx);
    const uint64_t uploadRetireSerial = ++m_uploadRetireSerial;
    if (!m_megaBuffers.isInitialized() && !m_megaBuffers.init(uploadCtx)) {
        return failUpload("SkiaVkMegaBuffers::init failed");
    }

    m_megaBuffers.setRetireSerial(uploadRetireSerial);
    m_megaBuffers.resetOffsets();
    m_instances.clear();
    m_instances.reserve(m_tessPlans.size());
    for (const auto& plan : m_tessPlans) {
        GPUPathInstance instance{};
        instance.fillColor[0] = plan.color[0];
        instance.fillColor[1] = plan.color[1];
        instance.fillColor[2] = plan.color[2];
        instance.fillColor[3] = plan.color[3];
        instance.shaderMatrixRow0[0] = 1.0f;
        instance.shaderMatrixRow0[1] = 0.0f;
        instance.shaderMatrixRow0[2] = 0.0f;
        instance.shaderMatrixRow0[3] = 0.0f;
        instance.shaderMatrixRow1[0] = 0.0f;
        instance.shaderMatrixRow1[1] = 1.0f;
        instance.shaderMatrixRow1[2] = 0.0f;
        instance.shaderMatrixRow1[3] = 0.0f;
        m_instances.push_back(instance);
    }
    uploadStats->setupInstancesMs = PerfElapsedMs(setupStartMs);

    const VkPipelineLayout pipelineLayout = pipelines ? pipelines->pipelineLayout() : VK_NULL_HANDLE;
    if (pipelineLayout != VK_NULL_HANDLE) {
        const double prepareStartMs = PerfNowMs();
        try {
            if (!pipelines || pipelines->executor() == nullptr) {
                return failUpload("PathMeshCopiedRuntime::uploadTessPlans: executor is null");
            }
            ScopedShaderModules ownedShaders(ctx.device);
            const VkShaderModule patchVert =
                    ownedShaders.add(createShaderModule(ctx.device, kPatchVertSpv, kPatchVertSpvSize));
            const VkShaderModule patchFanVert = ownedShaders.add(
                    createShaderModule(ctx.device, kPatchFanVertSpv, kPatchFanVertSpvSize));
            const VkShaderModule patchTesc =
                    ownedShaders.add(createShaderModule(ctx.device, kPatchTescSpv, kPatchTescSpvSize));
            const VkShaderModule patchTese =
                    ownedShaders.add(createShaderModule(ctx.device, kPatchTeseSpv, kPatchTeseSpvSize));
            const VkShaderModule patchFrag =
                    ownedShaders.add(createShaderModule(ctx.device, kPatchFragSpv, kPatchFragSpvSize));
            const VkShaderModule strokeVert =
                    ownedShaders.add(createShaderModule(ctx.device, kStrokeVertSpv, kStrokeVertSpvSize));
            const VkShaderModule strokeFrag =
                    ownedShaders.add(createShaderModule(ctx.device, kStrokeFragSpv, kStrokeFragSpvSize));
            const VkShaderModule bboxVert =
                    ownedShaders.add(createShaderModule(ctx.device, kBboxVertSpv, kBboxVertSpvSize));
            const VkShaderModule hullVert =
                    ownedShaders.add(createShaderModule(ctx.device, kHullVertSpv, kHullVertSpvSize));
            const VkShaderModule tessFillVert = ownedShaders.add(
                    createShaderModule(ctx.device, kTessFillVertSpv, kTessFillVertSpvSize));
            const VkShaderModule tessFillFrag = ownedShaders.add(
                    createShaderModule(ctx.device, kTessFillFragSpv, kTessFillFragSpvSize));

            VkShaderModule patchProbeVert = VK_NULL_HANDLE;
            VkShaderModule patchFanProbeVert = VK_NULL_HANDLE;
            VkShaderModule strokeProbeVert = VK_NULL_HANDLE;
            VkShaderModule bboxProbeVert = VK_NULL_HANDLE;
            VkShaderModule hullProbeVert = VK_NULL_HANDLE;
            VkShaderModule tessFillProbeVert = VK_NULL_HANDLE;
            if (probeModeEnabled) {
                patchProbeVert = ownedShaders.add(createShaderModule(
                        ctx.device, kPatchProbeVertSpv, kPatchProbeVertSpvSize));
                patchFanProbeVert = ownedShaders.add(createShaderModule(
                        ctx.device, kPatchFanProbeVertSpv, kPatchFanProbeVertSpvSize));
                strokeProbeVert = ownedShaders.add(createShaderModule(
                        ctx.device, kStrokeProbeVertSpv, kStrokeProbeVertSpvSize));
                bboxProbeVert = ownedShaders.add(
                        createShaderModule(ctx.device, kBboxProbeVertSpv, kBboxProbeVertSpvSize));
                hullProbeVert = ownedShaders.add(
                        createShaderModule(ctx.device, kHullProbeVertSpv, kHullProbeVertSpvSize));
                tessFillProbeVert = ownedShaders.add(createShaderModule(
                        ctx.device, kTessFillProbeVertSpv, kTessFillProbeVertSpvSize));
            }

            ExecutorShaderModules shaderModules{};
            shaderModules.patchVert = patchVert;
            shaderModules.patchProbeVert = patchProbeVert;
            shaderModules.patchFanVert = patchFanVert;
            shaderModules.patchFanProbeVert = patchFanProbeVert;
            shaderModules.patchTesc = patchTesc;
            shaderModules.patchTese = patchTese;
            shaderModules.patchFrag = patchFrag;
            shaderModules.strokeVert = strokeVert;
            shaderModules.strokeProbeVert = strokeProbeVert;
            shaderModules.strokeFrag = strokeFrag;
            shaderModules.bboxVert = bboxVert;
            shaderModules.bboxProbeVert = bboxProbeVert;
            shaderModules.hullVert = hullVert;
            shaderModules.hullProbeVert = hullProbeVert;
            shaderModules.tessFillVert = tessFillVert;
            shaderModules.tessFillProbeVert = tessFillProbeVert;
            shaderModules.tessFillFrag = tessFillFrag;
            shaderModules.useProbeShaderVariants = probeModeEnabled;

            if (demo_failpoint::ConsumeDemoFailPoint(
                        demo_failpoint::DemoFailPoint::kUploadPreparePipelines)) {
                return failUpload(MakeInjectedFailMessage(
                        "PathMeshCopiedRuntime/uploadTessPlans/preparePipelines"));
            }
            if (!pipelines->executor()->preparePipelines(
                        uploadCtx, m_tessPlans, m_instances, m_megaBuffers, pipelineLayout, shaderModules)) {
                return failUpload("SkiaVkTessPlanExecutor::preparePipelines failed");
            }
            m_lastPrepareStats = pipelines->replayStats();
            uploadStats->executorPrepareStats = m_lastPrepareStats;
        } catch (const std::exception& e) {
            return failUpload(std::string("PathMeshCopiedRuntime::uploadTessPlans: ") + e.what());
        }
        uploadStats->preparePipelinesMs = PerfElapsedMs(prepareStartMs);
    }

    const double transferStartMs = PerfNowMs();
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = ctx.commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer transferCmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(ctx.device, &allocInfo, &transferCmd) != VK_SUCCESS) {
        return failUpload("Failed to allocate Staging-to-Device transfer command buffer");
    }

    const auto freeTransferCmd = [&]() {
        if (transferCmd != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &transferCmd);
            transferCmd = VK_NULL_HANDLE;
        }
    };

    double transferSubstageStartMs = PerfNowMs();
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(transferCmd, &beginInfo) != VK_SUCCESS) {
        freeTransferCmd();
        return failUpload("Failed to begin Staging-to-Device transfer command buffer");
    }
    if (!m_megaBuffers.flushToDevice(transferCmd)) {
        freeTransferCmd();
        return failUpload("Failed to record Staging-to-Device transfer commands");
    }
    if (vkEndCommandBuffer(transferCmd) != VK_SUCCESS) {
        freeTransferCmd();
        return failUpload("Failed to end Staging-to-Device transfer command buffer");
    }
    uploadStats->transferRecordMs = PerfElapsedMs(transferSubstageStartMs);

    const auto recreateUploadFence = [&]() -> bool {
        if (m_uploadFence != VK_NULL_HANDLE) {
            vkDestroyFence(ctx.device, m_uploadFence, nullptr);
            m_uploadFence = VK_NULL_HANDLE;
        }
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        return vkCreateFence(ctx.device, &fenceInfo, nullptr, &m_uploadFence) == VK_SUCCESS;
    };
    if (m_uploadFence == VK_NULL_HANDLE && !recreateUploadFence()) {
        freeTransferCmd();
        return failUpload("Failed to create upload fence");
    }

    transferSubstageStartMs = PerfNowMs();
    if (vkWaitForFences(ctx.device, 1, &m_uploadFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        freeTransferCmd();
        return failUpload("Failed to wait for upload fence before transfer submit");
    }
    if (vkResetFences(ctx.device, 1, &m_uploadFence) != VK_SUCCESS) {
        freeTransferCmd();
        return failUpload("Failed to reset upload fence before transfer submit");
    }
    uploadStats->transferFenceReadyMs = PerfElapsedMs(transferSubstageStartMs);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &transferCmd;

    transferSubstageStartMs = PerfNowMs();
    if (vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, m_uploadFence) != VK_SUCCESS) {
        const bool fenceRestored = recreateUploadFence();
        freeTransferCmd();
        if (!fenceRestored) {
            return failUpload("Failed to submit transfer command buffer and recreate upload fence");
        }
        return failUpload("Failed to submit Staging-to-Device transfer command buffer");
    }
    uploadStats->transferSubmitMs = PerfElapsedMs(transferSubstageStartMs);

    transferSubstageStartMs = PerfNowMs();
    if (vkWaitForFences(ctx.device, 1, &m_uploadFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        freeTransferCmd();
        return failUpload("Failed to wait for upload fence after transfer submit");
    }
    uploadStats->transferWaitMs = PerfElapsedMs(transferSubstageStartMs);

    transferSubstageStartMs = PerfNowMs();
    m_completedUploadRetireSerial = uploadRetireSerial;
    freeTransferCmd();
    m_megaBuffers.releaseOldBuffers(m_completedUploadRetireSerial);
    uploadStats->transferCleanupMs = PerfElapsedMs(transferSubstageStartMs);
    uploadStats->transferTotalMs = PerfElapsedMs(transferStartMs);

    const double probeStartMs = PerfNowMs();
    if (probeModeEnabled) {
        std::string probeError;
        if (!rebuildProbeResources(ctx, pipelines, &probeError)) {
            return failUpload(probeError.empty() ? "RebuildProbeResources failed" : probeError);
        }
    } else {
        destroyProbeResources(ctx);
    }
    uploadStats->probeRebuildMs = PerfElapsedMs(probeStartMs);
    return true;
}

void PathMeshCopiedRuntime::resetUploadedTessGpuState(const PathMeshRuntimeContext& ctx) {
    destroyProbeResources(ctx);
    m_megaBuffers.resetOffsets();
    m_instances.clear();
    m_lastReplayRecordMs = 0.0;
}

void PathMeshCopiedRuntime::destroyTessPlans(const PathMeshRuntimeContext& ctx) {
    resetUploadedTessGpuState(ctx);
    m_tessPlans.clear();
}

bool PathMeshCopiedRuntime::waitForUploadIdle(const PathMeshRuntimeContext& ctx,
                                              std::string* error) const {
    if (ctx.device == VK_NULL_HANDLE || m_uploadFence == VK_NULL_HANDLE) {
        return true;
    }
    if (vkWaitForFences(ctx.device, 1, &m_uploadFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        SetError(error, "PathMeshCopiedRuntime::waitForUploadIdle: vkWaitForFences failed");
        return false;
    }
    return true;
}

bool PathMeshCopiedRuntime::clearProbeRecords(const PathMeshRuntimeContext& ctx,
                                              std::string* error) {
    if (m_probeMemory == VK_NULL_HANDLE) {
        SetError(error, "PathMeshCopiedRuntime::clearProbeRecords: probe memory is not initialized");
        return false;
    }
    if (ctx.device == VK_NULL_HANDLE) {
        SetError(error, "PathMeshCopiedRuntime::clearProbeRecords: device is not initialized");
        return false;
    }

    const size_t probeBufferSize =
            std::max<size_t>(m_probeRecordCount, 1) * sizeof(GpuProbeRecord);
    std::vector<uint8_t> zeroInit(probeBufferSize, 0);
    uploadBuffer(ctx.device, m_probeMemory, zeroInit.data(), zeroInit.size());
    return true;
}

bool PathMeshCopiedRuntime::readProbeRecords(const PathMeshRuntimeContext& ctx,
                                             std::vector<GpuProbeRecord>* outRecords,
                                             std::string* error) {
    if (!outRecords) {
        SetError(error, "PathMeshCopiedRuntime::readProbeRecords: outRecords is null");
        return false;
    }

    outRecords->clear();
    if (m_probeMemory == VK_NULL_HANDLE) {
        SetError(error, "PathMeshCopiedRuntime::readProbeRecords: probe memory is not initialized");
        return false;
    }
    if (ctx.device == VK_NULL_HANDLE) {
        SetError(error, "PathMeshCopiedRuntime::readProbeRecords: device is not initialized");
        return false;
    }

    outRecords->resize(m_probeRecordCount);
    if (m_probeRecordCount == 0) {
        return true;
    }

    const VkDeviceSize readSize =
            static_cast<VkDeviceSize>(m_probeRecordCount * sizeof(GpuProbeRecord));
    void* mapped = nullptr;
    if (vkMapMemory(ctx.device, m_probeMemory, 0, readSize, 0, &mapped) != VK_SUCCESS) {
        outRecords->clear();
        SetError(error, "PathMeshCopiedRuntime::readProbeRecords: vkMapMemory failed");
        return false;
    }

    std::memcpy(outRecords->data(), mapped, static_cast<size_t>(readSize));
    vkUnmapMemory(ctx.device, m_probeMemory);
    return true;
}

void PathMeshCopiedRuntime::recordDrawCommands(VkCommandBuffer cmd,
                                               PathMeshWindowPipelineBundle* pipelines,
                                               VkExtent2D swapchainExtent,
                                               bool probeModeEnabled) {
    const VkPipelineLayout pipelineLayout = pipelines ? pipelines->pipelineLayout() : VK_NULL_HANDLE;
    if (pipelineLayout == VK_NULL_HANDLE || !pipelines || pipelines->executor() == nullptr ||
        !hasReplayDrawData()) {
        m_lastReplayRecordMs = 0.0;
        return;
    }

    const double replayStartMs = PerfNowMs();
    const float viewportWidth =
            static_cast<float>(swapchainExtent.width > 0 ? swapchainExtent.width : 1);
    const float viewportHeight =
            static_cast<float>(swapchainExtent.height > 0 ? swapchainExtent.height : 1);
    TessPushConstants pushConstants{};
    pushConstants.viewportScaleX = 2.0f / viewportWidth;
    pushConstants.viewportScaleY = 2.0f / viewportHeight;
    pushConstants.maxResolveLevel = static_cast<float>(skgpu::tess::kMaxResolveLevel);
    pushConstants.baseCmdIndex = 0;

    if (probeModeEnabled && m_probeDescriptorSet != VK_NULL_HANDLE) {
        pipelines->executor()->execute(
                cmd, m_megaBuffers, pipelineLayout, pushConstants, m_probeDescriptorSet, &m_probePassInfos);
        m_lastReplayRecordMs = PerfElapsedMs(replayStartMs);
        return;
    }

    pipelines->executor()->execute(cmd, m_megaBuffers, pipelineLayout, pushConstants);
    m_lastReplayRecordMs = PerfElapsedMs(replayStartMs);
}

bool PathMeshCopiedRuntime::hasReplayDrawData() const {
    if (m_instances.empty()) {
        return false;
    }
    const bool hasIndirect = m_megaBuffers.getIndirectCmdBuffer() != VK_NULL_HANDLE &&
                             m_megaBuffers.getIndirectCmdCount() > 0;
    const bool hasIndexedIndirect =
            m_megaBuffers.getIndexedIndirectCmdBuffer() != VK_NULL_HANDLE &&
            m_megaBuffers.getIndexedIndirectCmdCount() > 0;
    return hasIndirect || hasIndexedIndirect;
}

VulkanUploadContext PathMeshCopiedRuntime::makeUploadContext(
        const PathMeshRuntimeContext& ctx) const {
    VulkanUploadContext uploadCtx;
    uploadCtx.instance = ctx.instance;
    uploadCtx.physicalDevice = ctx.physicalDevice;
    uploadCtx.device = ctx.device;
    uploadCtx.queue = ctx.graphicsQueue;
    uploadCtx.commandPool = ctx.commandPool;
    return uploadCtx;
}

uint32_t PathMeshCopiedRuntime::findMemoryType(const PathMeshRuntimeContext& ctx,
                                               uint32_t typeBits,
                                               VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties memoryProps{};
    vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &memoryProps);

    for (uint32_t i = 0; i < memoryProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            ((memoryProps.memoryTypes[i].propertyFlags & props) == props)) {
            return i;
        }
    }
    throw std::runtime_error("No suitable memory type");
}

void PathMeshCopiedRuntime::createBuffer(const PathMeshRuntimeContext& ctx,
                                         VkDeviceSize size,
                                         VkBufferUsageFlags usage,
                                         VkMemoryPropertyFlags props,
                                         VkBuffer& outBuffer,
                                         VkDeviceMemory& outMemory) const {
    outBuffer = VK_NULL_HANDLE;
    outMemory = VK_NULL_HANDLE;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx.device, &bufferInfo, nullptr, &outBuffer) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateBuffer failed");
    }

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(ctx.device, outBuffer, &memoryRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memoryRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx, memoryRequirements.memoryTypeBits, props);
    if (vkAllocateMemory(ctx.device, &allocInfo, nullptr, &outMemory) != VK_SUCCESS) {
        vkDestroyBuffer(ctx.device, outBuffer, nullptr);
        outBuffer = VK_NULL_HANDLE;
        throw std::runtime_error("vkAllocateMemory failed");
    }

    if (vkBindBufferMemory(ctx.device, outBuffer, outMemory, 0) != VK_SUCCESS) {
        vkFreeMemory(ctx.device, outMemory, nullptr);
        vkDestroyBuffer(ctx.device, outBuffer, nullptr);
        outBuffer = VK_NULL_HANDLE;
        outMemory = VK_NULL_HANDLE;
        throw std::runtime_error("vkBindBufferMemory failed");
    }
}

void PathMeshCopiedRuntime::uploadBuffer(VkDevice device,
                                         VkDeviceMemory memory,
                                         const void* src,
                                         size_t size) const {
    void* dst = nullptr;
    if (vkMapMemory(device, memory, 0, size, 0, &dst) != VK_SUCCESS) {
        throw std::runtime_error("vkMapMemory failed");
    }
    std::memcpy(dst, src, size);
    vkUnmapMemory(device, memory);
}

void PathMeshCopiedRuntime::destroyBuffer(VkDevice device,
                                          VkBuffer& buffer,
                                          VkDeviceMemory& memory) const {
    if (device != VK_NULL_HANDLE && buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer, nullptr);
    }
    if (device != VK_NULL_HANDLE && memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, memory, nullptr);
    }
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
}

VkShaderModule PathMeshCopiedRuntime::createShaderModule(VkDevice device,
                                                         const uint32_t* data,
                                                         size_t sizeBytes) const {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = sizeBytes;
    createInfo.pCode = data;

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateShaderModule failed");
    }
    return module;
}

bool PathMeshCopiedRuntime::rebuildProbeResources(const PathMeshRuntimeContext& ctx,
                                                  PathMeshWindowPipelineBundle* pipelines,
                                                  std::string* error) {
    destroyProbeResources(ctx);

    std::vector<std::vector<TessProbePassInfo>> probePassInfos;
    size_t probeRecordCount = 0;
    if (demo_failpoint::ConsumeDemoFailPoint(demo_failpoint::DemoFailPoint::kProbeBuildLayout)) {
        SetError(error, MakeInjectedFailMessage(
                                "PathMeshCopiedRuntime/rebuildProbeResources/BuildProbeLayout"));
        return false;
    }
    if (!BuildProbeLayout(m_tessPlans, &probePassInfos, &probeRecordCount, error)) {
        return false;
    }

    if (!pipelines || pipelines->executor() == nullptr) {
        SetError(error, "PathMeshCopiedRuntime::rebuildProbeResources: executor is not initialized");
        return false;
    }

    const VkDescriptorSetLayout probeLayout = pipelines->getProbeDescriptorSetLayout();
    if (probeLayout == VK_NULL_HANDLE) {
        SetError(error,
                 "PathMeshCopiedRuntime::rebuildProbeResources: probe descriptor set layout is null");
        return false;
    }

    const size_t probeBufferSize = std::max<size_t>(probeRecordCount, 1) * sizeof(GpuProbeRecord);
    std::vector<uint8_t> zeroInit(probeBufferSize, 0);
    try {
        if (demo_failpoint::ConsumeDemoFailPoint(
                    demo_failpoint::DemoFailPoint::kProbeCreateBuffer)) {
            throw std::runtime_error(MakeInjectedFailMessage(
                    "PathMeshCopiedRuntime/rebuildProbeResources/CreateBuffer"));
        }
        createBuffer(ctx,
                     static_cast<VkDeviceSize>(probeBufferSize),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     m_probeBuffer,
                     m_probeMemory);
        uploadBuffer(ctx.device, m_probeMemory, zeroInit.data(), zeroInit.size());
    } catch (const std::exception& e) {
        destroyProbeResources(ctx);
        SetError(error, std::string("PathMeshCopiedRuntime::rebuildProbeResources: ") + e.what());
        return false;
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;
    if (demo_failpoint::ConsumeDemoFailPoint(
                demo_failpoint::DemoFailPoint::kProbeDescriptorPool)) {
        destroyProbeResources(ctx);
        SetError(error,
                 MakeInjectedFailMessage(
                         "PathMeshCopiedRuntime/rebuildProbeResources/vkCreateDescriptorPool"));
        return false;
    }
    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &m_probeDescriptorPool) !=
        VK_SUCCESS) {
        destroyProbeResources(ctx);
        SetError(error, "PathMeshCopiedRuntime::rebuildProbeResources: vkCreateDescriptorPool failed");
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_probeDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &probeLayout;
    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &m_probeDescriptorSet) != VK_SUCCESS) {
        destroyProbeResources(ctx);
        SetError(error, "PathMeshCopiedRuntime::rebuildProbeResources: vkAllocateDescriptorSets failed");
        return false;
    }

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = m_probeBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = static_cast<VkDeviceSize>(probeBufferSize);

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_probeDescriptorSet;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);

    m_probePassInfos = std::move(probePassInfos);
    m_probeRecordCount = probeRecordCount;
    return true;
}

void PathMeshCopiedRuntime::destroyProbeResources(const PathMeshRuntimeContext& ctx) {
    m_probePassInfos.clear();
    m_probeRecordCount = 0;
    m_probeDescriptorSet = VK_NULL_HANDLE;

    if (ctx.device != VK_NULL_HANDLE && m_probeDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx.device, m_probeDescriptorPool, nullptr);
    }
    m_probeDescriptorPool = VK_NULL_HANDLE;
    destroyBuffer(ctx.device, m_probeBuffer, m_probeMemory);
}

}  // namespace skia_port
