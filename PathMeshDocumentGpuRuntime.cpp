#include "PathMeshDocumentGpuRuntime.h"

#include "SkiaProbeLayout.h"
#include <ported_skia/src/gpu/tessellate/Tessellation.h>

#include <algorithm>

namespace skia_port {

namespace {

void SetError(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

void FillShaderMatrixRows(const Mat3& m, float row0[4], float row1[4]) {
    row0[0] = m.v[0];
    row0[1] = m.v[1];
    row0[2] = m.v[2];
    row0[3] = 0.0f;
    row1[0] = m.v[3];
    row1[1] = m.v[4];
    row1[2] = m.v[5];
    row1[3] = 0.0f;
}

bool AreFloatArraysEqual(const float* lhs, const float* rhs, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

bool AreMat3Equal(const Mat3& lhs, const Mat3& rhs) {
    return AreFloatArraysEqual(lhs.v, rhs.v, 9);
}

bool AreRectFsEqual(const RectF& lhs, const RectF& rhs) {
    return lhs.left == rhs.left && lhs.top == rhs.top && lhs.right == rhs.right &&
           lhs.bottom == rhs.bottom;
}

bool AreSceneTransformsEqual(const SceneViewTransform& lhs, const SceneViewTransform& rhs) {
    return AreMat3Equal(lhs.pathToNdc, rhs.pathToNdc) &&
           AreMat3Equal(lhs.pathToDevice, rhs.pathToDevice) &&
           AreRectFsEqual(lhs.deviceClipBounds, rhs.deviceClipBounds);
}

PathMeshDocumentGpuRuntime::ReplayGpuUploadViewKey BuildReplayGpuUploadViewKey(
        const SceneViewTransform& sceneTransform, bool strictProbeBatching) {
    PathMeshDocumentGpuRuntime::ReplayGpuUploadViewKey viewKey;
    viewKey.pathToDevice = sceneTransform.pathToDevice;
    viewKey.strictProbeBatching = strictProbeBatching;
    return viewKey;
}

bool AreTrianglesEqual(const Triangle& lhs, const Triangle& rhs) {
    return lhs.p0.x == rhs.p0.x && lhs.p0.y == rhs.p0.y && lhs.p1.x == rhs.p1.x &&
           lhs.p1.y == rhs.p1.y && lhs.p2.x == rhs.p2.x && lhs.p2.y == rhs.p2.y;
}

bool ArePrepareOptionsEqual(const PatchPrepareOptions& lhs, const PatchPrepareOptions& rhs) {
    if (!AreFloatArraysEqual(lhs.viewMatrix.v, rhs.viewMatrix.v, 9) ||
        !AreFloatArraysEqual(lhs.shaderMatrix.v, rhs.shaderMatrix.v, 9) ||
        !AreFloatArraysEqual(lhs.pathMatrix.v, rhs.pathMatrix.v, 9) ||
        !AreRectFsEqual(lhs.clipConservativeBounds, rhs.clipConservativeBounds) ||
        lhs.hasClipConservativeBounds != rhs.hasClipConservativeBounds ||
        lhs.infinitySupport != rhs.infinitySupport ||
        lhs.vertexIDSupport != rhs.vertexIDSupport ||
        lhs.stencilOnly != rhs.stencilOnly ||
        lhs.wireframe != rhs.wireframe ||
        lhs.hasStencilClip != rhs.hasStencilClip ||
        lhs.preChopCurvesIfNecessary != rhs.preChopCurvesIfNecessary ||
        lhs.extraTriangles.size() != rhs.extraTriangles.size()) {
        return false;
    }

    for (size_t i = 0; i < lhs.extraTriangles.size(); ++i) {
        if (!AreTrianglesEqual(lhs.extraTriangles[i], rhs.extraTriangles[i])) {
            return false;
        }
    }
    return true;
}

bool AreReplayInputKeysEqual(const DocumentReplayInputKey& lhs, const DocumentReplayInputKey& rhs) {
    return lhs.documentRevision == rhs.documentRevision && lhs.msaaSamples == rhs.msaaSamples;
}

bool AreReplayCaptureKeysEqual(const DocumentReplayCaptureKey& lhs,
                               const DocumentReplayCaptureKey& rhs) {
    return AreReplayInputKeysEqual(lhs.replayInputKey, rhs.replayInputKey) &&
           ArePrepareOptionsEqual(lhs.capturePrepareOptions, rhs.capturePrepareOptions);
}

bool AreReplayGpuUploadViewKeysEqual(
        const PathMeshDocumentGpuRuntime::ReplayGpuUploadViewKey& lhs,
        const PathMeshDocumentGpuRuntime::ReplayGpuUploadViewKey& rhs) {
    return lhs.strictProbeBatching == rhs.strictProbeBatching &&
           AreMat3Equal(lhs.pathToDevice, rhs.pathToDevice);
}

bool AreUploadOffsetsEqual(const VkTessPassUploadOffsets& lhs,
                           const VkTessPassUploadOffsets& rhs) {
    return lhs.vertexByteOffset == rhs.vertexByteOffset &&
           lhs.indexByteOffset == rhs.indexByteOffset &&
           lhs.instanceByteOffset == rhs.instanceByteOffset;
}

uint32_t AlignByteOffset(uint32_t offset, uint32_t strideBytes) {
    if (strideBytes == 0) {
        return offset;
    }
    const uint32_t remainder = offset % strideBytes;
    if (remainder == 0) {
        return offset;
    }
    return offset + (strideBytes - remainder);
}

struct GeometryUploadOffsetState {
    uint32_t triangleVertexByteOffset = 0;
    uint32_t triangleIndexByteOffset = 0;
    uint32_t patchInstanceByteOffset = 0;
};

void SimulateAppendTessPassUpload(const VkTessPassUploadBytes& upload,
                                  GeometryUploadOffsetState* state,
                                  VkTessPassUploadOffsets* outOffsets) {
    if (!state || !outOffsets) {
        return;
    }

    *outOffsets = {};

    if (!upload.vertexBytes.empty()) {
        state->triangleVertexByteOffset =
                AlignByteOffset(state->triangleVertexByteOffset, upload.vertexStrideBytes);
        outOffsets->vertexByteOffset = state->triangleVertexByteOffset;
        state->triangleVertexByteOffset += static_cast<uint32_t>(upload.vertexBytes.size());
    }

    if (!upload.indexBytes.empty()) {
        outOffsets->indexByteOffset = state->triangleIndexByteOffset;
        state->triangleIndexByteOffset += static_cast<uint32_t>(upload.indexBytes.size());
    }

    if (!upload.instanceBytes.empty()) {
        state->patchInstanceByteOffset =
                AlignByteOffset(state->patchInstanceByteOffset, upload.instanceStrideBytes);
        outOffsets->instanceByteOffset = state->patchInstanceByteOffset;
        state->patchInstanceByteOffset += static_cast<uint32_t>(upload.instanceBytes.size());
    }
}

bool IsValidRect(const RectF& rect) {
    return rect.right > rect.left && rect.bottom > rect.top;
}

RectF SelectReplayBounds(const TessCapturePlan& plan) {
    if (IsValidRect(plan.inputPath.tightBounds)) {
        return plan.inputPath.tightBounds;
    }
    return plan.inputPath.bounds;
}

bool RectsOverlap(const RectF& a, const RectF& b) {
    return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
}

enum class ReplayPhaseClass : uint8_t {
    kStencilPrelude,
    kColorResolve,
};

ReplayPhaseClass PhaseClassForPass(TessPlanPassKind passKind) {
    switch (passKind) {
        case TessPlanPassKind::kStencilCurvePatches:
        case TessPlanPassKind::kStencilWedgePatches:
        case TessPlanPassKind::kStencilFanTriangles:
            return ReplayPhaseClass::kStencilPrelude;
        case TessPlanPassKind::kFillFanTriangles:
        case TessPlanPassKind::kCoverHulls:
        case TessPlanPassKind::kCoverBoundingBoxes:
        case TessPlanPassKind::kStrokePatches:
        case TessPlanPassKind::kUnknown:
        default:
            return ReplayPhaseClass::kColorResolve;
    }
}

size_t PassKindIndex(TessPlanPassKind passKind) {
    return static_cast<size_t>(passKind);
}

bool StencilFaceEq(const TessStencilFaceInfo& a, const TessStencilFaceInfo& b) {
    return a.ref == b.ref && a.test == b.test && a.testMask == b.testMask &&
           a.passOp == b.passOp && a.failOp == b.failOp && a.writeMask == b.writeMask;
}

bool ProgramInfoEq(const TessProgramInfo& a, const TessProgramInfo& b) {
    return a.kind == b.kind && a.primitiveType == b.primitiveType &&
           a.stencilSettings.kind == b.stencilSettings.kind &&
           a.stencilSettings.usesStencil == b.stencilSettings.usesStencil &&
           a.stencilSettings.writesStencil == b.stencilSettings.writesStencil &&
           a.stencilSettings.resetsStencil == b.stencilSettings.resetsStencil &&
           a.stencilSettings.testsStencil == b.stencilSettings.testsStencil &&
           a.stencilSettings.usesClipBit == b.stencilSettings.usesClipBit &&
           a.stencilSettings.twoSided == b.stencilSettings.twoSided &&
           StencilFaceEq(a.stencilSettings.front, b.stencilSettings.front) &&
           StencilFaceEq(a.stencilSettings.back, b.stencilSettings.back) &&
           a.layout.kind == b.layout.kind &&
           a.layout.vertexStrideBytes == b.layout.vertexStrideBytes &&
           a.layout.instanceStrideBytes == b.layout.instanceStrideBytes &&
           a.patchAttribMask == b.patchAttribMask &&
           a.usesVertexID == b.usesVertexID &&
           a.wireframe == b.wireframe &&
           a.stencilOnly == b.stencilOnly &&
           a.stroke.enabled == b.stroke.enabled &&
           a.stroke.hasDynamicStroke == b.stroke.hasDynamicStroke &&
           a.stroke.hairline == b.stroke.hairline &&
           a.stroke.numRadialSegmentsPerRadian == b.stroke.numRadialSegmentsPerRadian &&
           a.stroke.joinType == b.stroke.joinType &&
           a.stroke.strokeRadius == b.stroke.strokeRadius;
}

struct ReplayBatchDrawMeta {
    TessProgramInfo programInfo;
    TessPlanPassKind passKind = TessPlanPassKind::kUnknown;
    bool isIndexed = false;
    uint32_t shapeIndex = 0;
    uint32_t passIndex = 0;
    bool hasCustomMaxResolveLevel = false;
    float maxResolveLevel = 0.0f;
    uint32_t baseSSBOIndex = 0;
    uint32_t firstCmdIndex = 0;
};

bool CanAppendReplayBatch(const ReplayBatchDrawMeta& prev,
                          const ReplayBatchDrawMeta& next,
                          bool strictProbeBatching) {
    if (!ProgramInfoEq(prev.programInfo, next.programInfo) ||
        prev.isIndexed != next.isIndexed ||
        prev.hasCustomMaxResolveLevel != next.hasCustomMaxResolveLevel) {
        return false;
    }

    if (next.hasCustomMaxResolveLevel && prev.maxResolveLevel != next.maxResolveLevel) {
        return false;
    }

    if (strictProbeBatching &&
        (prev.shapeIndex != next.shapeIndex || prev.passIndex != next.passIndex)) {
        return false;
    }

    if (prev.passKind != next.passKind) {
        return false;
    }
    if (prev.baseSSBOIndex + 1 != next.baseSSBOIndex) {
        return false;
    }
    if (prev.firstCmdIndex + 1 != next.firstCmdIndex) {
        return false;
    }

    return true;
}

const PatchBufferData::Chunk* FindPatchChunkForDraw(const PatchBufferData& patch,
                                                    const TessDrawCommand& drawCmd) {
    if (patch.patchStrideBytes == 0 || drawCmd.instanceCount == 0) {
        return nullptr;
    }

    const uint64_t drawStart = static_cast<uint64_t>(drawCmd.baseInstance);
    const uint64_t drawEnd = drawStart + static_cast<uint64_t>(drawCmd.instanceCount);
    for (const PatchBufferData::Chunk& chunk : patch.chunks) {
        const uint64_t chunkStart = static_cast<uint64_t>(chunk.basePatch);
        const uint64_t chunkEnd = chunkStart + static_cast<uint64_t>(chunk.patchCount);
        if (drawStart >= chunkStart && drawEnd <= chunkEnd) {
            return &chunk;
        }
    }

    return nullptr;
}

uint32_t ComputeRebasedPatchFirstInstanceRelative(uint32_t strideBytes,
                                                  const PatchBufferData::Chunk& chunk,
                                                  const TessDrawCommand& drawCmd) {
    if (strideBytes == 0) {
        return 0;
    }

    const uint64_t chunkUploadBase = static_cast<uint64_t>(chunk.byteOffset) / strideBytes;
    const uint64_t rebasedBaseInstance =
            static_cast<uint64_t>(drawCmd.baseInstance) - static_cast<uint64_t>(chunk.basePatch);
    return static_cast<uint32_t>(chunkUploadBase + rebasedBaseInstance);
}

uint32_t CalcReplayStrokeFlags(const TessProgramInfo& info) {
    uint32_t strokeFlags = 0;
    if (info.usesVertexID) {
        strokeFlags |= 1u;
    }
    if ((info.patchAttribMask &
         static_cast<uint32_t>(skgpu::tess::PatchAttribs::kExplicitCurveType)) != 0) {
        strokeFlags |= 2u;
    }
    if (info.stroke.hairline) {
        strokeFlags |= 4u;
    }
    return strokeFlags;
}

uint32_t CalcReplayProbeFlagsTemplate(const TessProgramInfo& info) {
    uint32_t probeFlags = 0;
    if (info.layout.kind == TessLayoutKind::kMiddleOut &&
        info.primitiveType == TessPrimitiveType::kTriangles) {
        probeFlags |= 2u;
    }
    return probeFlags;
}

}  // namespace

void PathMeshDocumentGpuRuntime::Cleanup(const PathMeshRuntimeContext& ctx) {
    for (auto& uploadEntry : m_replayGpuUploads) {
        uploadEntry.gpuState.Cleanup(ctx);
    }
    m_replayGpuUploads.clear();
    m_geometryGpuState.Cleanup(ctx);
    Reset();
}

void PathMeshDocumentGpuRuntime::Reset() {
    m_documentRevision = 0;
    m_sceneSourceKind = PathMeshSceneSourceKind::kBuiltin;
    m_fillShapeCount = 0;
    m_strokePathCount = 0;
    m_hasSceneBounds = false;
    m_sceneBounds = {};
    ResetCapturedTessPlans();
}

void PathMeshDocumentGpuRuntime::SyncDocumentMetadata(const PathMeshDocumentRuntime& document) {
    SyncDocumentScene(document, nullptr, nullptr);
}

bool PathMeshDocumentGpuRuntime::SyncDocumentScene(const PathMeshDocumentRuntime& document,
                                                   const RectF* overrideSceneBounds,
                                                   bool* outChanged) {
    RectF nextBounds{};
    bool hasBounds = false;
    if (overrideSceneBounds) {
        nextBounds = *overrideSceneBounds;
        hasBounds = true;
    } else {
        hasBounds = document.TryGetSceneBounds(&nextBounds);
        if (!hasBounds) {
            nextBounds = {};
        }
    }

    const uint64_t nextRevision = document.SceneRevision();
    const PathMeshSceneSourceKind nextSceneSourceKind = document.SceneSourceKind();
    const size_t nextFillShapeCount = document.SvgFillShapes().size();
    const size_t nextStrokePathCount = document.SvgStrokePaths().size();

    const bool changed = m_documentRevision != nextRevision ||
                         m_sceneSourceKind != nextSceneSourceKind ||
                         m_fillShapeCount != nextFillShapeCount ||
                         m_strokePathCount != nextStrokePathCount ||
                         m_hasSceneBounds != hasBounds ||
                         (hasBounds &&
                          (m_sceneBounds.left != nextBounds.left ||
                           m_sceneBounds.top != nextBounds.top ||
                           m_sceneBounds.right != nextBounds.right ||
                           m_sceneBounds.bottom != nextBounds.bottom));
    if (outChanged) {
        *outChanged = changed;
    }
    if (changed) {
        ResetCapturedTessPlans();
    }

    m_documentRevision = nextRevision;
    m_sceneSourceKind = nextSceneSourceKind;
    m_fillShapeCount = nextFillShapeCount;
    m_strokePathCount = nextStrokePathCount;
    m_hasSceneBounds = hasBounds;
    m_sceneBounds = nextBounds;
    return true;
}

bool PathMeshDocumentGpuRuntime::CaptureTessPlans(const DocumentReplayInputKey& replayInputKey,
                                                  const PatchPrepareOptions& capturePrepareOptions,
                                                  const std::vector<SvgFillShape>& fillShapes,
                                                  const std::vector<SvgStrokePath>& strokePaths,
                                                  std::string* error) {
    if (replayInputKey.documentRevision != m_documentRevision) {
        SetError(error, "CaptureTessPlans: replay input key does not match current document revision");
        return false;
    }
    if (IsCapturedFor(replayInputKey, capturePrepareOptions)) {
        return true;
    }
    ResetCapturedTessPlans();

    const AAMode aaType =
            (replayInputKey.msaaSamples != VK_SAMPLE_COUNT_1_BIT) ? AAMode::kMSAA : AAMode::kNone;

    auto captureShape = [&](const SkPath& path,
                            const SvgNodeMetadata& nodeMeta,
                            const float color[4],
                            const PathDrawOptions& drawOpts) -> bool {
        TessCapturePlan plan;
        std::string localError;
        if (!CapturePathDrawPlanOriginalSkia(path, drawOpts, &plan, &localError)) {
            ResetCapturedTessPlans();
            SetError(error, "CapturePathDrawPlanOriginalSkia failed: " + localError);
            return false;
        }

        plan.node.nodeTag = nodeMeta.nodeTag;
        plan.node.nodeId = nodeMeta.nodeId;
        plan.node.nodeIndex = nodeMeta.nodeIndex;
        plan.node.contourIndex = nodeMeta.contourIndex;
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
        drawOpts.patchOptions = capturePrepareOptions;
        const float color[4] = {shape.r, shape.g, shape.b, shape.a};
        if (!captureShape(shape.skPath, shape.node, color, drawOpts)) {
            return false;
        }
    }

    for (const auto& stroke : strokePaths) {
        PathDrawOptions drawOpts;
        drawOpts.isStroke = true;
        drawOpts.aaType = aaType;
        drawOpts.patchOptions = capturePrepareOptions;

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
        if (!captureShape(stroke.skPath, stroke.node, color, drawOpts)) {
            return false;
        }
    }

    m_capturedDocumentRevision = m_documentRevision;
    m_captureKey.replayInputKey = replayInputKey;
    m_captureKey.capturePrepareOptions = capturePrepareOptions;
    rebuildReplayInstances();
    return true;
}

void PathMeshDocumentGpuRuntime::ResetCapturedTessPlans() {
    ResetSharedGeometryGpuUploadState();
    ResetReplayGpuUploadStates();
    m_capturedDocumentRevision = 0;
    m_tessPlans.clear();
    m_captureKey = {};
    m_replayInstances.clear();
    m_replayPassUploadsKey = {};
    m_replayPassUploads.clear();
    m_replayGeometryUploadOffsetsKey = {};
    m_replayGeometryUploadOffsets.clear();
    m_replayPassDescriptorsKey = {};
    m_replayPassDescriptors.clear();
    m_replayDrawDescriptorsKey = {};
    m_replayDrawDescriptors.clear();
    m_replayPrepareStatsKey = {};
    m_replayPrepareStats = {};
    m_replayOrdersKey = {};
    m_replayOrders = {};
    m_replayBatchPlansKey = {};
    m_replayBatchPlans = {};
    m_sharedGeometryUploadCount = 0;
    m_sharedGeometryReuseCount = 0;
    m_probeLayoutDocumentRevision = 0;
    m_probeRecordCount = 0;
    m_probePassInfos.clear();
}

bool PathMeshDocumentGpuRuntime::IsCurrentFor(const PathMeshDocumentRuntime& document) const {
    return m_documentRevision != 0 && m_documentRevision == document.SceneRevision();
}

bool PathMeshDocumentGpuRuntime::IsCapturedFor(const DocumentReplayInputKey& replayInputKey,
                                               const PatchPrepareOptions& capturePrepareOptions) const {
    DocumentReplayCaptureKey key;
    key.replayInputKey = replayInputKey;
    key.capturePrepareOptions = capturePrepareOptions;
    return HasCapturedTessPlans() && AreReplayCaptureKeysEqual(m_captureKey, key);
}

bool PathMeshDocumentGpuRuntime::HasCapturedTessPlans() const {
    return m_capturedDocumentRevision != 0 && m_capturedDocumentRevision == m_documentRevision &&
           !m_tessPlans.empty();
}

bool PathMeshDocumentGpuRuntime::HasReplayInstances() const {
    return HasCapturedTessPlans() && m_replayInstances.size() == m_tessPlans.size();
}

bool PathMeshDocumentGpuRuntime::EnsureReplayPassUploads(std::string* error) {
    if (!HasCapturedTessPlans()) {
        m_replayPassUploadsKey = {};
        m_replayPassUploads.clear();
        m_replayPrepareStatsKey = {};
        m_replayPrepareStats = {};
        return true;
    }
    if (HasReplayPassUploads()) {
        return true;
    }

    std::vector<std::vector<VkTessPassUploadBytes>> passUploads;
    ExecutorCachedReplayPrepareStats prepareStats;
    prepareStats.planCount = static_cast<uint32_t>(m_tessPlans.size());
    passUploads.resize(m_tessPlans.size());
    for (size_t shapeIndex = 0; shapeIndex < m_tessPlans.size(); ++shapeIndex) {
        const TessCapturePlan& plan = m_tessPlans[shapeIndex];
        passUploads[shapeIndex].resize(plan.passes.size());
        for (size_t passIndex = 0; passIndex < plan.passes.size(); ++passIndex) {
            const TessPassPlan& pass = plan.passes[passIndex];
            std::string localError;
            if (!BuildTessPassUploadBytes(
                        pass, &passUploads[shapeIndex][passIndex], &localError)) {
                m_replayPassUploadsKey = {};
                m_replayPassUploads.clear();
                m_replayPrepareStatsKey = {};
                m_replayPrepareStats = {};
                SetError(error,
                         "EnsureReplayPassUploads failed for shape " + std::to_string(shapeIndex) +
                                 " pass " + std::to_string(passIndex) + ": " + localError);
                return false;
            }

            const size_t passKindIndex = PassKindIndex(pass.kind);
            ++prepareStats.passCount;
            ++prepareStats.passCountByKind[passKindIndex];
            prepareStats.drawCmdCount += static_cast<uint32_t>(pass.drawCommands.size());
            prepareStats.drawCmdCountByKind[passKindIndex] +=
                    static_cast<uint32_t>(pass.drawCommands.size());

            const VkTessPassUploadBytes& uploadBytes = passUploads[shapeIndex][passIndex];
            prepareStats.vertexUploadBytes += uploadBytes.vertexBytes.size();
            prepareStats.indexUploadBytes += uploadBytes.indexBytes.size();
            prepareStats.instanceUploadBytes += uploadBytes.instanceBytes.size();

            for (const TessDrawCommand& drawCmd : pass.drawCommands) {
                ++prepareStats.globalInstanceCount;
                if (drawCmd.kind == TessDrawCommandKind::kIndexedInstanced) {
                    ++prepareStats.indexedIndirectCmdCount;
                } else {
                    ++prepareStats.indirectCmdCount;
                }
            }
        }
    }

    m_replayPassUploadsKey = m_captureKey;
    m_replayPassUploads = std::move(passUploads);
    m_replayPrepareStatsKey = m_captureKey;
    m_replayPrepareStats = prepareStats;
    return true;
}

bool PathMeshDocumentGpuRuntime::HasReplayPassUploads() const {
    return HasCapturedTessPlans() &&
           AreReplayCaptureKeysEqual(m_replayPassUploadsKey, m_captureKey) &&
           m_replayPassUploads.size() == m_tessPlans.size();
}

bool PathMeshDocumentGpuRuntime::EnsureReplayGeometryUploadOffsets(std::string* error) {
    (void)error;
    if (!HasCapturedTessPlans()) {
        m_replayGeometryUploadOffsetsKey = {};
        m_replayGeometryUploadOffsets.clear();
        return true;
    }
    if (HasReplayGeometryUploadOffsets()) {
        return true;
    }
    if (!EnsureReplayPassUploads(error)) {
        m_replayGeometryUploadOffsetsKey = {};
        m_replayGeometryUploadOffsets.clear();
        return false;
    }

    GeometryUploadOffsetState state{};
    std::vector<std::vector<VkTessPassUploadOffsets>> uploadOffsets;
    uploadOffsets.resize(m_replayPassUploads.size());
    for (size_t shapeIndex = 0; shapeIndex < m_replayPassUploads.size(); ++shapeIndex) {
        const auto& shapePassUploads = m_replayPassUploads[shapeIndex];
        auto& shapePassOffsets = uploadOffsets[shapeIndex];
        shapePassOffsets.resize(shapePassUploads.size());
        for (size_t passIndex = 0; passIndex < shapePassUploads.size(); ++passIndex) {
            SimulateAppendTessPassUpload(
                    shapePassUploads[passIndex], &state, &shapePassOffsets[passIndex]);
        }
    }

    m_replayGeometryUploadOffsetsKey = m_captureKey;
    m_replayGeometryUploadOffsets = std::move(uploadOffsets);
    return true;
}

bool PathMeshDocumentGpuRuntime::HasReplayGeometryUploadOffsets() const {
    return HasCapturedTessPlans() &&
           AreReplayCaptureKeysEqual(m_replayGeometryUploadOffsetsKey, m_captureKey) &&
           m_replayGeometryUploadOffsets.size() == m_tessPlans.size();
}

bool PathMeshDocumentGpuRuntime::EnsureReplayPassDescriptors(std::string* error) {
    if (!HasCapturedTessPlans()) {
        m_replayPassDescriptorsKey = {};
        m_replayPassDescriptors.clear();
        return true;
    }
    if (HasReplayPassDescriptors()) {
        return true;
    }
    if (!HasReplayInstances()) {
        m_replayPassDescriptorsKey = {};
        m_replayPassDescriptors.clear();
        SetError(error,
                 "EnsureReplayPassDescriptors: replay instances are not prepared");
        return false;
    }

    std::vector<std::vector<ExecutorReplayPassDescriptor>> passDescriptors;
    passDescriptors.resize(m_tessPlans.size());
    for (size_t shapeIndex = 0; shapeIndex < m_tessPlans.size(); ++shapeIndex) {
        const TessCapturePlan& plan = m_tessPlans[shapeIndex];
        const GPUPathInstance& baseInstance = m_replayInstances[shapeIndex];
        passDescriptors[shapeIndex].resize(plan.passes.size());
        for (size_t passIndex = 0; passIndex < plan.passes.size(); ++passIndex) {
            const TessPassPlan& pass = plan.passes[passIndex];
            ExecutorReplayPassDescriptor descriptor;
            descriptor.programInfo = pass.programInfo;
            descriptor.passKind = pass.kind;
            descriptor.probeFlagsTemplate = CalcReplayProbeFlagsTemplate(pass.programInfo);
            descriptor.instanceTemplate = baseInstance;
            FillShaderMatrixRows(
                    pass.shaderMatrix,
                    descriptor.instanceTemplate.shaderMatrixRow0,
                    descriptor.instanceTemplate.shaderMatrixRow1);
            descriptor.instanceTemplate.strokeTessArgs[0] =
                    pass.programInfo.stroke.numRadialSegmentsPerRadian;
            descriptor.instanceTemplate.strokeTessArgs[1] = pass.programInfo.stroke.joinType;
            descriptor.instanceTemplate.strokeTessArgs[2] = pass.programInfo.stroke.strokeRadius;
            descriptor.instanceTemplate.strokeTessArgs[3] =
                    static_cast<float>(CalcReplayStrokeFlags(pass.programInfo));

            descriptor.hasCustomMaxResolveLevel = pass.patchBuffer.patchCount > 0;
            if (descriptor.hasCustomMaxResolveLevel) {
                const float maxAllowed = static_cast<float>(skgpu::tess::kMaxResolveLevel);
                descriptor.maxResolveLevel = std::clamp(
                        static_cast<float>(pass.patchBuffer.requiredResolveLevel), 0.0f, maxAllowed);
            }
            passDescriptors[shapeIndex][passIndex] = descriptor;
        }
    }

    m_replayPassDescriptorsKey = m_captureKey;
    m_replayPassDescriptors = std::move(passDescriptors);
    return true;
}

bool PathMeshDocumentGpuRuntime::HasReplayPassDescriptors() const {
    return HasCapturedTessPlans() &&
           AreReplayCaptureKeysEqual(m_replayPassDescriptorsKey, m_captureKey) &&
           m_replayPassDescriptors.size() == m_tessPlans.size();
}

bool PathMeshDocumentGpuRuntime::EnsureReplayDrawDescriptors(std::string* error) {
    if (!HasCapturedTessPlans()) {
        m_replayDrawDescriptorsKey = {};
        m_replayDrawDescriptors.clear();
        return true;
    }
    if (HasReplayDrawDescriptors()) {
        return true;
    }
    if (!EnsureReplayPassUploads(error)) {
        m_replayDrawDescriptorsKey = {};
        m_replayDrawDescriptors.clear();
        return false;
    }
    if (!EnsureReplayPassDescriptors(error)) {
        m_replayDrawDescriptorsKey = {};
        m_replayDrawDescriptors.clear();
        return false;
    }

    std::vector<std::vector<std::vector<ExecutorReplayDrawDescriptor>>> drawDescriptors;
    drawDescriptors.resize(m_tessPlans.size());

    uint32_t nextGlobalInstanceOrdinal = 0;
    uint32_t nextIndirectCmdIndex = 0;
    uint32_t nextIndexedCmdIndex = 0;
    for (size_t shapeIndex = 0; shapeIndex < m_tessPlans.size(); ++shapeIndex) {
        const TessCapturePlan& plan = m_tessPlans[shapeIndex];
        drawDescriptors[shapeIndex].resize(plan.passes.size());
        for (size_t passIndex = 0; passIndex < plan.passes.size(); ++passIndex) {
            const TessPassPlan& pass = plan.passes[passIndex];
            const VkTessPassUploadBytes& uploadBytes = m_replayPassUploads[shapeIndex][passIndex];
            const bool isFanPass = uploadBytes.route == VkTessPassUploadRoute::kTriangles;
            const bool isBBoxPass = uploadBytes.route == VkTessPassUploadRoute::kBoundingBox;
            const uint32_t patchStrideBytes = uploadBytes.instanceStrideBytes == 0
                                                      ? 1
                                                      : uploadBytes.instanceStrideBytes;

            uint32_t localTriangleVertexCount = 0;
            uint32_t bboxInstanceByteCursor = 0;

            auto& passDrawDescriptors = drawDescriptors[shapeIndex][passIndex];
            passDrawDescriptors.resize(pass.drawCommands.size());
            for (size_t drawIndex = 0; drawIndex < pass.drawCommands.size(); ++drawIndex) {
                const TessDrawCommand& drawCmd = pass.drawCommands[drawIndex];
                ExecutorReplayDrawDescriptor descriptor;
                descriptor.isIndexed = (drawCmd.kind == TessDrawCommandKind::kIndexedInstanced);
                descriptor.elementCount = drawCmd.elementCount;
                descriptor.instanceCount = drawCmd.instanceCount;
                descriptor.globalInstanceOrdinal = nextGlobalInstanceOrdinal++;
                descriptor.commandStreamIndex =
                        descriptor.isIndexed ? nextIndexedCmdIndex++ : nextIndirectCmdIndex++;

                if (isFanPass) {
                    descriptor.usesPatchBaseForFirstInstance = false;
                    descriptor.firstInstanceOffset = 0;
                    if (descriptor.isIndexed) {
                        descriptor.firstIndexOffset = 0;
                        descriptor.vertexOffset = static_cast<int32_t>(localTriangleVertexCount);
                    } else {
                        descriptor.firstVertexOffset = localTriangleVertexCount;
                    }
                } else {
                    descriptor.usesPatchBaseForFirstInstance = true;
                    if (descriptor.isIndexed) {
                        descriptor.firstIndexOffset = drawCmd.baseIndex;
                        descriptor.vertexOffset = static_cast<int32_t>(drawCmd.baseVertex);
                    } else {
                        descriptor.firstVertexOffset = drawCmd.baseVertex;
                    }

                    if (isBBoxPass) {
                        descriptor.firstInstanceOffset = bboxInstanceByteCursor / patchStrideBytes;
                    } else {
                        const PatchBufferData::Chunk* chunk =
                                FindPatchChunkForDraw(pass.patchBuffer, drawCmd);
                        if (!chunk && drawIndex < pass.patchBuffer.chunks.size()) {
                            chunk = &pass.patchBuffer.chunks[drawIndex];
                        }
                        if (chunk) {
                            descriptor.firstInstanceOffset = ComputeRebasedPatchFirstInstanceRelative(
                                    patchStrideBytes, *chunk, drawCmd);
                        } else {
                            descriptor.firstInstanceOffset = drawCmd.baseInstance;
                        }
                    }
                }

                passDrawDescriptors[drawIndex] = descriptor;

                if (isFanPass) {
                    localTriangleVertexCount += drawCmd.elementCount;
                }
                if (isBBoxPass && drawCmd.instanceCount > 0) {
                    bboxInstanceByteCursor += drawCmd.instanceCount * patchStrideBytes;
                }
            }
        }
    }

    m_replayDrawDescriptorsKey = m_captureKey;
    m_replayDrawDescriptors = std::move(drawDescriptors);
    return true;
}

bool PathMeshDocumentGpuRuntime::HasReplayDrawDescriptors() const {
    return HasCapturedTessPlans() &&
           AreReplayCaptureKeysEqual(m_replayDrawDescriptorsKey, m_captureKey) &&
           m_replayDrawDescriptors.size() == m_tessPlans.size();
}

bool PathMeshDocumentGpuRuntime::HasReplayPrepareStats() const {
    return HasCapturedTessPlans() &&
           AreReplayCaptureKeysEqual(m_replayPrepareStatsKey, m_captureKey);
}

bool PathMeshDocumentGpuRuntime::HasReplayGpuUpload() const {
    if (HasSharedGeometryGpuUpload()) {
        return true;
    }
    for (const auto& uploadEntry : m_replayGpuUploads) {
        if (uploadEntry.uploadedCaptureKey.replayInputKey.documentRevision != 0) {
            return true;
        }
    }
    return false;
}

bool PathMeshDocumentGpuRuntime::HasSharedGeometryGpuUpload() const {
    return HasCapturedTessPlans() &&
           AreReplayCaptureKeysEqual(m_sharedGeometryUploadedKey, m_captureKey);
}

bool PathMeshDocumentGpuRuntime::HasSharedGeometryGpuUploadForCurrentInput() const {
    return HasSharedGeometryGpuUpload();
}

bool PathMeshDocumentGpuRuntime::EnsureSharedGeometryGpuUploaded(
        const PathMeshRuntimeContext& ctx,
        PathMeshUploadStats* stats,
        std::string* error) {
    if (!HasCapturedTessPlans()) {
        ResetSharedGeometryGpuUploadState();
        return true;
    }
    if (HasSharedGeometryGpuUploadForCurrentInput()) {
        ++m_sharedGeometryReuseCount;
        if (stats) {
            stats->reusedSharedGeometry = true;
            stats->sharedGeometryUploadCount = m_sharedGeometryUploadCount;
            stats->sharedGeometryReuseCount = m_sharedGeometryReuseCount;
        }
        return true;
    }
    if (!uploadSharedGeometry(ctx, stats, error)) {
        return false;
    }
    m_sharedGeometryUploadedKey = m_captureKey;
    ++m_sharedGeometryUploadCount;
    if (stats) {
        stats->reusedSharedGeometry = false;
        stats->sharedGeometryUploadCount = m_sharedGeometryUploadCount;
        stats->sharedGeometryReuseCount = m_sharedGeometryReuseCount;
    }
    return true;
}

void PathMeshDocumentGpuRuntime::ResetSharedGeometryGpuUploadState() {
    m_geometryGpuState.ResetUploadedState();
    m_sharedGeometryUploadedKey = {};
}

bool PathMeshDocumentGpuRuntime::HasReplayGpuUploadForCurrentInput(
        const SceneViewTransform& sceneTransform, bool strictProbeBatching) const {
    const ReplayGpuUploadEntry* uploadEntry =
            findReplayGpuUploadEntry(sceneTransform, strictProbeBatching);
    return HasSharedGeometryGpuUploadForCurrentInput() && uploadEntry &&
           AreReplayCaptureKeysEqual(uploadEntry->uploadedCaptureKey, m_captureKey);
}

void PathMeshDocumentGpuRuntime::ResetReplayGpuUploadState(const SceneViewTransform& sceneTransform,
                                                           bool strictProbeBatching) {
    ReplayGpuUploadEntry* uploadEntry = findReplayGpuUploadEntry(sceneTransform, strictProbeBatching);
    if (!uploadEntry) {
        return;
    }
    uploadEntry->gpuState.ResetUploadedState();
    uploadEntry->uploadedCaptureKey = {};
    uploadEntry->uploadStats = {};
}

void PathMeshDocumentGpuRuntime::ResetReplayGpuUploadStates() {
    for (auto& uploadEntry : m_replayGpuUploads) {
        uploadEntry.gpuState.ResetUploadedState();
        uploadEntry.uploadedCaptureKey = {};
        uploadEntry.uploadStats = {};
    }
}

void PathMeshDocumentGpuRuntime::NoteReplayGpuUploadReuse(const SceneViewTransform& sceneTransform,
                                                          bool strictProbeBatching,
                                                          PathMeshUploadStats* stats) {
    ReplayGpuUploadEntry& uploadEntry =
            ensureReplayGpuUploadEntry(sceneTransform, strictProbeBatching);
    ++uploadEntry.reuseHitCount;
    fillReplayGpuUploadStats(uploadEntry, false, stats);
}

void PathMeshDocumentGpuRuntime::MarkReplayGpuUploaded(const SceneViewTransform& sceneTransform,
                                                       bool strictProbeBatching,
                                                       PathMeshUploadStats* stats) {
    ReplayGpuUploadEntry& uploadEntry =
            ensureReplayGpuUploadEntry(sceneTransform, strictProbeBatching);
    uploadEntry.uploadedCaptureKey = m_captureKey;
    ++uploadEntry.uploadCount;
    if (stats) {
        uploadEntry.uploadStats = *stats;
    } else {
        uploadEntry.uploadStats = {};
    }
    fillReplayGpuUploadStats(uploadEntry, false, stats);
}

PathMeshDocumentReplayGpuState& PathMeshDocumentGpuRuntime::ReplayGpuState(
        const SceneViewTransform& sceneTransform,
        bool strictProbeBatching,
        PathMeshUploadStats* stats) {
    bool createdNewSlot = false;
    ReplayGpuUploadEntry& uploadEntry =
            ensureReplayGpuUploadEntry(sceneTransform, strictProbeBatching, &createdNewSlot);
    ++uploadEntry.useCount;
    fillReplayGpuUploadStats(uploadEntry, createdNewSlot, stats);
    return uploadEntry.gpuState;
}

const PathMeshDocumentReplayGpuState* PathMeshDocumentGpuRuntime::FindReplayGpuState(
        const SceneViewTransform& sceneTransform, bool strictProbeBatching) const {
    const ReplayGpuUploadEntry* uploadEntry =
            findReplayGpuUploadEntry(sceneTransform, strictProbeBatching);
    return uploadEntry ? &uploadEntry->gpuState : nullptr;
}

bool PathMeshDocumentGpuRuntime::EnsureReplayOrders(std::string* error) {
    (void)error;
    if (!HasCapturedTessPlans()) {
        m_replayOrdersKey = {};
        m_replayOrders = {};
        return true;
    }
    if (HasReplayOrders()) {
        return true;
    }

    struct ShapePhaseRefs {
        ReplayPhaseClass phaseClass = ReplayPhaseClass::kColorResolve;
        std::vector<ExecutorReplayDrawRef> draws;
    };
    struct ShapeReplayRefs {
        RectF bounds{};
        bool hasBounds = false;
        std::vector<ShapePhaseRefs> phases;
    };

    std::vector<ShapeReplayRefs> preparedShapes;
    preparedShapes.reserve(m_tessPlans.size());
    ExecutorCachedReplayOrders replayOrders;

    for (size_t shapeIndex = 0; shapeIndex < m_tessPlans.size(); ++shapeIndex) {
        const TessCapturePlan& plan = m_tessPlans[shapeIndex];
        preparedShapes.push_back({});
        ShapeReplayRefs& preparedShape = preparedShapes.back();
        preparedShape.bounds = SelectReplayBounds(plan);
        preparedShape.hasBounds = IsValidRect(preparedShape.bounds);

        for (size_t passIndex = 0; passIndex < plan.passes.size(); ++passIndex) {
            const TessPassPlan& pass = plan.passes[passIndex];
            const ReplayPhaseClass phaseClass = PhaseClassForPass(pass.kind);
            if (preparedShape.phases.empty() ||
                preparedShape.phases.back().phaseClass != phaseClass) {
                preparedShape.phases.push_back({});
                preparedShape.phases.back().phaseClass = phaseClass;
            }
            ShapePhaseRefs& phaseRefs = preparedShape.phases.back();
            for (size_t drawIndex = 0; drawIndex < pass.drawCommands.size(); ++drawIndex) {
                ExecutorReplayDrawRef drawRef;
                drawRef.shapeIndex = static_cast<uint32_t>(shapeIndex);
                drawRef.passIndex = static_cast<uint32_t>(passIndex);
                drawRef.drawIndex = static_cast<uint32_t>(drawIndex);
                drawRef.passKind = pass.kind;
                phaseRefs.draws.push_back(drawRef);
                replayOrders.originalDrawRefs.push_back(drawRef);
            }
        }
    }

    auto canJoinPhaseWindow = [&](size_t begin, size_t endExclusive, size_t candidateIndex) {
        if (candidateIndex >= preparedShapes.size()) {
            return false;
        }
        const ShapeReplayRefs& candidate = preparedShapes[candidateIndex];
        if (!candidate.hasBounds) {
            return false;
        }
        for (size_t shapeIndex = begin; shapeIndex < endExclusive; ++shapeIndex) {
            const ShapeReplayRefs& existing = preparedShapes[shapeIndex];
            if (!existing.hasBounds || RectsOverlap(existing.bounds, candidate.bounds)) {
                return false;
            }
        }
        return true;
    };

    size_t shapeIndex = 0;
    while (shapeIndex < preparedShapes.size()) {
        size_t windowEnd = shapeIndex + 1;
        while (windowEnd < preparedShapes.size() &&
               canJoinPhaseWindow(shapeIndex, windowEnd, windowEnd)) {
            ++windowEnd;
        }

        if (windowEnd - shapeIndex <= 1) {
            const ShapeReplayRefs& preparedShape = preparedShapes[shapeIndex];
            for (const ShapePhaseRefs& phase : preparedShape.phases) {
                replayOrders.windowedDrawRefs.insert(replayOrders.windowedDrawRefs.end(),
                                                     phase.draws.begin(),
                                                     phase.draws.end());
            }
            ++shapeIndex;
            continue;
        }

        size_t maxPhaseCount = 0;
        for (size_t windowShape = shapeIndex; windowShape < windowEnd; ++windowShape) {
            maxPhaseCount = std::max(maxPhaseCount, preparedShapes[windowShape].phases.size());
        }

        for (size_t phaseIndex = 0; phaseIndex < maxPhaseCount; ++phaseIndex) {
            for (ReplayPhaseClass phaseClass :
                 {ReplayPhaseClass::kStencilPrelude, ReplayPhaseClass::kColorResolve}) {
                for (size_t windowShape = shapeIndex; windowShape < windowEnd; ++windowShape) {
                    const ShapeReplayRefs& preparedShape = preparedShapes[windowShape];
                    if (phaseIndex >= preparedShape.phases.size()) {
                        continue;
                    }
                    const ShapePhaseRefs& phase = preparedShape.phases[phaseIndex];
                    if (phase.phaseClass != phaseClass) {
                        continue;
                    }
                    replayOrders.windowedDrawRefs.insert(replayOrders.windowedDrawRefs.end(),
                                                         phase.draws.begin(),
                                                         phase.draws.end());
                }
            }
        }

        shapeIndex = windowEnd;
    }

    m_replayOrdersKey = m_captureKey;
    m_replayOrders = std::move(replayOrders);
    return true;
}

bool PathMeshDocumentGpuRuntime::HasReplayOrders() const {
    return HasCapturedTessPlans() &&
           AreReplayCaptureKeysEqual(m_replayOrdersKey, m_captureKey) &&
           m_replayOrders.originalDrawRefs.size() == m_replayOrders.windowedDrawRefs.size();
}

bool PathMeshDocumentGpuRuntime::EnsureReplayBatchPlans(std::string* error) {
    if (!HasCapturedTessPlans()) {
        m_replayBatchPlansKey = {};
        m_replayBatchPlans = {};
        return true;
    }
    if (HasReplayBatchPlans()) {
        return true;
    }
    if (!EnsureReplayOrders(error)) {
        m_replayBatchPlansKey = {};
        m_replayBatchPlans = {};
        return false;
    }
    if (!EnsureReplayPassDescriptors(error)) {
        m_replayBatchPlansKey = {};
        m_replayBatchPlans = {};
        return false;
    }
    if (!EnsureReplayDrawDescriptors(error)) {
        m_replayBatchPlansKey = {};
        m_replayBatchPlans = {};
        return false;
    }

    std::vector<std::vector<std::vector<ReplayBatchDrawMeta>>> drawMetaLookup;
    drawMetaLookup.resize(m_tessPlans.size());
    for (size_t shapeIndex = 0; shapeIndex < m_tessPlans.size(); ++shapeIndex) {
        const TessCapturePlan& plan = m_tessPlans[shapeIndex];
        drawMetaLookup[shapeIndex].resize(plan.passes.size());
        for (size_t passIndex = 0; passIndex < plan.passes.size(); ++passIndex) {
            const ExecutorReplayPassDescriptor& descriptor =
                    m_replayPassDescriptors[shapeIndex][passIndex];
            const auto& passDrawDescriptors = m_replayDrawDescriptors[shapeIndex][passIndex];
            auto& drawMetas = drawMetaLookup[shapeIndex][passIndex];
            drawMetas.resize(passDrawDescriptors.size());

            for (size_t drawIndex = 0; drawIndex < passDrawDescriptors.size(); ++drawIndex) {
                const ExecutorReplayDrawDescriptor& drawDescriptor = passDrawDescriptors[drawIndex];
                ReplayBatchDrawMeta meta;
                meta.programInfo = descriptor.programInfo;
                meta.passKind = descriptor.passKind;
                meta.isIndexed = drawDescriptor.isIndexed;
                meta.shapeIndex = static_cast<uint32_t>(shapeIndex);
                meta.passIndex = static_cast<uint32_t>(passIndex);
                meta.hasCustomMaxResolveLevel = descriptor.hasCustomMaxResolveLevel;
                meta.maxResolveLevel = descriptor.maxResolveLevel;
                meta.baseSSBOIndex = drawDescriptor.globalInstanceOrdinal;
                meta.firstCmdIndex = drawDescriptor.commandStreamIndex;
                drawMetas[drawIndex] = meta;
            }
        }
    }

    auto lookupDrawMeta = [&](const ExecutorReplayDrawRef& drawRef,
                              const ReplayBatchDrawMeta** outMeta) -> bool {
        if (drawRef.shapeIndex >= drawMetaLookup.size()) {
            SetError(error, "EnsureReplayBatchPlans: draw ref shape index out of range");
            return false;
        }
        const auto& passDraws = drawMetaLookup[drawRef.shapeIndex];
        if (drawRef.passIndex >= passDraws.size()) {
            SetError(error, "EnsureReplayBatchPlans: draw ref pass index out of range");
            return false;
        }
        const auto& drawMetas = passDraws[drawRef.passIndex];
        if (drawRef.drawIndex >= drawMetas.size()) {
            SetError(error, "EnsureReplayBatchPlans: draw ref draw index out of range");
            return false;
        }
        *outMeta = &drawMetas[drawRef.drawIndex];
        return true;
    };

    auto buildBatchSeeds = [&](const std::vector<ExecutorReplayDrawRef>& drawRefs,
                               bool strictProbeBatching,
                               std::vector<ExecutorReplayBatchSeed>* outSeeds) -> bool {
        outSeeds->clear();
        if (drawRefs.empty()) {
            return true;
        }

        const ReplayBatchDrawMeta* prevMeta = nullptr;
        if (!lookupDrawMeta(drawRefs.front(), &prevMeta)) {
            return false;
        }

        size_t batchBegin = 0;
        for (size_t drawRefIndex = 1; drawRefIndex < drawRefs.size(); ++drawRefIndex) {
            const ReplayBatchDrawMeta* currentMeta = nullptr;
            if (!lookupDrawMeta(drawRefs[drawRefIndex], &currentMeta)) {
                return false;
            }
            if (!CanAppendReplayBatch(*prevMeta, *currentMeta, strictProbeBatching)) {
                ExecutorReplayBatchSeed batchSeed;
                batchSeed.firstDrawRefIndex = static_cast<uint32_t>(batchBegin);
                batchSeed.drawCount = static_cast<uint32_t>(drawRefIndex - batchBegin);
                outSeeds->push_back(batchSeed);
                batchBegin = drawRefIndex;
            }
            prevMeta = currentMeta;
        }

        ExecutorReplayBatchSeed batchSeed;
        batchSeed.firstDrawRefIndex = static_cast<uint32_t>(batchBegin);
        batchSeed.drawCount = static_cast<uint32_t>(drawRefs.size() - batchBegin);
        outSeeds->push_back(batchSeed);
        return true;
    };

    ExecutorCachedReplayBatchPlans replayBatchPlans;
    if (!buildBatchSeeds(m_replayOrders.originalDrawRefs, true, &replayBatchPlans.originalBatches) ||
        !buildBatchSeeds(
                m_replayOrders.windowedDrawRefs, false, &replayBatchPlans.windowedBatches)) {
        m_replayBatchPlansKey = {};
        m_replayBatchPlans = {};
        return false;
    }

    m_replayBatchPlansKey = m_captureKey;
    m_replayBatchPlans = std::move(replayBatchPlans);
    return true;
}

bool PathMeshDocumentGpuRuntime::HasReplayBatchPlans() const {
    return HasCapturedTessPlans() && HasReplayOrders() &&
           AreReplayCaptureKeysEqual(m_replayBatchPlansKey, m_captureKey);
}

bool PathMeshDocumentGpuRuntime::EnsureProbeLayout(std::string* error) {
    if (!HasCapturedTessPlans()) {
        m_probeLayoutDocumentRevision = 0;
        m_probeRecordCount = 0;
        m_probePassInfos.clear();
        return true;
    }
    if (m_probeLayoutDocumentRevision == m_capturedDocumentRevision) {
        return true;
    }

    std::vector<std::vector<TessProbePassInfo>> probePassInfos;
    size_t probeRecordCount = 0;
    if (!BuildProbeLayout(m_tessPlans, &probePassInfos, &probeRecordCount, error)) {
        m_probeLayoutDocumentRevision = 0;
        m_probeRecordCount = 0;
        m_probePassInfos.clear();
        return false;
    }

    m_probeLayoutDocumentRevision = m_capturedDocumentRevision;
    m_probeRecordCount = probeRecordCount;
    m_probePassInfos = std::move(probePassInfos);
    return true;
}

bool PathMeshDocumentGpuRuntime::HasProbeLayout() const {
    return m_probeLayoutDocumentRevision != 0 &&
           m_probeLayoutDocumentRevision == m_capturedDocumentRevision;
}

PathMeshDocumentGpuRuntime::ReplayGpuUploadEntry*
PathMeshDocumentGpuRuntime::findReplayGpuUploadEntry(const SceneViewTransform& sceneTransform,
                                                     bool strictProbeBatching) {
    const ReplayGpuUploadViewKey viewKey =
            BuildReplayGpuUploadViewKey(sceneTransform, strictProbeBatching);
    for (auto& uploadEntry : m_replayGpuUploads) {
        if (AreReplayGpuUploadViewKeysEqual(uploadEntry.viewKey, viewKey)) {
            return &uploadEntry;
        }
    }
    return nullptr;
}

const PathMeshDocumentGpuRuntime::ReplayGpuUploadEntry*
PathMeshDocumentGpuRuntime::findReplayGpuUploadEntry(const SceneViewTransform& sceneTransform,
                                                     bool strictProbeBatching) const {
    const ReplayGpuUploadViewKey viewKey =
            BuildReplayGpuUploadViewKey(sceneTransform, strictProbeBatching);
    for (const auto& uploadEntry : m_replayGpuUploads) {
        if (AreReplayGpuUploadViewKeysEqual(uploadEntry.viewKey, viewKey)) {
            return &uploadEntry;
        }
    }
    return nullptr;
}

PathMeshDocumentGpuRuntime::ReplayGpuUploadEntry&
PathMeshDocumentGpuRuntime::ensureReplayGpuUploadEntry(const SceneViewTransform& sceneTransform,
                                                       bool strictProbeBatching,
                                                       bool* outCreatedNewSlot) {
    if (ReplayGpuUploadEntry* existing =
                findReplayGpuUploadEntry(sceneTransform, strictProbeBatching)) {
        if (outCreatedNewSlot) {
            *outCreatedNewSlot = false;
        }
        return *existing;
    }

    ReplayGpuUploadEntry uploadEntry;
    uploadEntry.viewKey = BuildReplayGpuUploadViewKey(sceneTransform, strictProbeBatching);
    uploadEntry.slotId = m_nextReplayGpuUploadSlotId++;
    m_replayGpuUploads.push_back(std::move(uploadEntry));
    if (outCreatedNewSlot) {
        *outCreatedNewSlot = true;
    }
    return m_replayGpuUploads.back();
}

void PathMeshDocumentGpuRuntime::fillReplayGpuUploadStats(const ReplayGpuUploadEntry& uploadEntry,
                                                          bool createdNewSlot,
                                                          PathMeshUploadStats* stats) const {
    if (!stats) {
        return;
    }
    stats->replayGpuStrictProbeBatching = uploadEntry.viewKey.strictProbeBatching;
    stats->replayGpuSlotCreated = stats->replayGpuSlotCreated || createdNewSlot;
    stats->replayGpuSlotId = uploadEntry.slotId;
    stats->replayGpuSlotUseCount = uploadEntry.useCount;
    stats->replayGpuSlotUploadCount = uploadEntry.uploadCount;
    stats->replayGpuSlotReuseHitCount = uploadEntry.reuseHitCount;
    stats->replayGpuSlotCount = m_replayGpuUploads.size();
}

bool PathMeshDocumentGpuRuntime::uploadSharedGeometry(const PathMeshRuntimeContext& ctx,
                                                      PathMeshUploadStats* stats,
                                                      std::string* error) {
    if (!EnsureReplayPassUploads(error)) {
        return false;
    }
    if (!EnsureReplayGeometryUploadOffsets(error)) {
        return false;
    }

    if (!m_geometryGpuState.BeginUpload(ctx, error)) {
        return false;
    }

    auto failUpload = [&](const std::string& message) {
        m_geometryGpuState.ResetUploadedState();
        SetError(error, message);
        return false;
    };

    SkiaVkMegaBuffers& geometryBuffers = m_geometryGpuState.MegaBuffers();
    for (size_t shapeIndex = 0; shapeIndex < m_replayPassUploads.size(); ++shapeIndex) {
        const auto& shapePassUploads = m_replayPassUploads[shapeIndex];
        const auto& shapePassOffsets = m_replayGeometryUploadOffsets[shapeIndex];
        if (shapePassUploads.size() != shapePassOffsets.size()) {
            return failUpload(
                    "uploadSharedGeometry: cached geometry upload offsets size mismatch");
        }
        for (size_t passIndex = 0; passIndex < shapePassUploads.size(); ++passIndex) {
            VkTessPassUploadOffsets actualOffsets;
            if (!geometryBuffers.appendTessPassUpload(shapePassUploads[passIndex], &actualOffsets)) {
                return failUpload("uploadSharedGeometry: appendTessPassUpload failed");
            }
            if (!AreUploadOffsetsEqual(actualOffsets, shapePassOffsets[passIndex])) {
                return failUpload(
                        "uploadSharedGeometry: simulated geometry upload offsets diverged");
            }
        }
    }

    if (!m_geometryGpuState.SubmitPendingUpload(ctx, stats, error)) {
        m_geometryGpuState.ResetUploadedState();
        return false;
    }
    return true;
}

void PathMeshDocumentGpuRuntime::rebuildReplayInstances() {
    m_replayInstances.clear();
    m_replayInstances.reserve(m_tessPlans.size());
    for (const auto& plan : m_tessPlans) {
        GPUPathInstance instance{};
        instance.fillColor[0] = plan.color[0];
        instance.fillColor[1] = plan.color[1];
        instance.fillColor[2] = plan.color[2];
        instance.fillColor[3] = plan.color[3];
        instance.shaderMatrixRow0[0] = 1.0f;
        instance.shaderMatrixRow1[1] = 1.0f;
        m_replayInstances.push_back(instance);
    }
}

bool PathMeshDocumentGpuRuntime::TryGetSceneBounds(RectF* outBounds) const {
    if (!outBounds || !m_hasSceneBounds) {
        return false;
    }
    *outBounds = m_sceneBounds;
    return true;
}

}  // namespace skia_port
