#include "SkiaCpuMeshDrawTarget.h"

#include "SkiaCpuGpuBuffers.h"
#include "SkiaCpuOpFlushState.h"
#include "SkiaCpuPrePrepare.h"
#include "SkiaTessellatorAccess.h"
#include <ported_skia/include/core/SkMatrix.h>
#include <ported_skia/include/core/SkPoint.h>
#include <ported_skia/include/core/SkRect.h>
#include <ported_skia/include/core/SkRefCnt.h>
#include <ported_skia/src/base/SkArenaAlloc.h>
#include <ported_skia/src/gpu/ResourceKey.h>
#include <ported_skia/src/gpu/BufferWriter.h>
#include <ported_skia/src/gpu/ganesh/GrBuffer.h>
#include <ported_skia/src/gpu/ganesh/GrCaps.h>
#include <ported_skia/src/gpu/ganesh/GrDstProxyView.h>
#include <ported_skia/src/gpu/ganesh/GrEagerVertexAllocator.h>
#include <ported_skia/src/gpu/ganesh/GrGpuBuffer.h>
#include <ported_skia/src/gpu/ganesh/GrMeshDrawTarget.h>
#include <ported_skia/src/gpu/ganesh/GrRenderTargetProxy.h>
#include <ported_skia/src/gpu/ganesh/GrResourceProvider.h>
#include <ported_skia/src/gpu/ganesh/GrSurfaceProxyView.h>
#include <ported_skia/src/gpu/ganesh/GrVertexChunkArray.h>
#include <ported_skia/src/gpu/ganesh/geometry/GrInnerFanTriangulator.h>
#include <ported_skia/src/gpu/ganesh/geometry/GrTriangulator.h>
#include <ported_skia/src/gpu/ganesh/GrShaderCaps.h>
#include <ported_skia/src/gpu/ganesh/tessellate/PathTessellator.h>
#include <ported_skia/src/gpu/tessellate/AffineMatrix.h>
#include <ported_skia/src/gpu/tessellate/FixedCountBufferUtils.h>
#include <ported_skia/src/gpu/tessellate/MiddleOutPolygonTriangulator.h>
#include <ported_skia/src/gpu/tessellate/Tessellation.h>
#include <ported_skia/src/gpu/ganesh/tessellate/StrokeTessellator.h>
#include <ported_skia/include/core/SkStrokeRec.h>
#include <ported_skia/include/core/SkPaint.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace skia_port {

namespace {

using PatchAttribs = skgpu::tess::PatchAttribs;

static GrCaps make_caps(bool infinitySupport, bool vertexIDSupport) {
    GrShaderCaps shaderCaps;
    shaderCaps.fInfinitySupport = infinitySupport;
    shaderCaps.fVertexIDSupport = vertexIDSupport;
    return GrCaps(shaderCaps);
}

static void set_error(std::string* error, const std::string& text) {
    if (error) {
        *error = text;
    }
}

static SkMatrix to_sk_matrix(const Mat3& m) {
    SkScalar values[9] = {
        m.v[0], m.v[1], m.v[2],
        m.v[3], m.v[4], m.v[5],
        m.v[6], m.v[7], m.v[8],
    };
    SkMatrix out;
    out.set9(values);
    return out;
}

static SkRect to_sk_rect(const RectF& r) {
    return SkRect::MakeLTRB(r.left, r.top, r.right, r.bottom);
}

static sk_sp<CpuGpuBuffer> make_dynamic_cpu_buffer(size_t byteSize, GrGpuBufferType type) {
    return sk_make_sp<CpuGpuBuffer>(byteSize, type, kDynamic_GrAccessPattern);
}

static void trim_buffer_tail(CpuGpuBuffer* buffer, size_t bytesToTrim) {
    if (!buffer) {
        return;
    }
    if (bytesToTrim > buffer->size()) {
        buffer->resize(0);
        return;
    }
    buffer->resize(buffer->size() - bytesToTrim);
}

class CpuMeshDrawTarget final : public GrMeshDrawTarget {
public:
    explicit CpuMeshDrawTarget(const GrCaps& caps)
            : fAllocator(1024 * 1024)
            , fCaps(caps)
            , fWriteView(&fDummyProxy)
            , fAppliedClip(SkISize::Make(1 << 29, 1 << 29)) {}

    void recordDraw(const GrGeometryProcessor*,
                    const GrSimpleMesh[],
                    int,
                    const GrSurfaceProxy* const[],
                    GrPrimitiveType) override {}

    void* makeVertexSpace(size_t vertexSize, int vertexCount, sk_sp<const GrBuffer>* buffer,
                          int* startVertex) override {
        int actualVertexCount = 0;
        return this->makeVertexSpaceAtLeast(vertexSize,
                                            vertexCount,
                                            vertexCount,
                                            buffer,
                                            startVertex,
                                            &actualVertexCount);
    }

    uint16_t* makeIndexSpace(int indexCount, sk_sp<const GrBuffer>* buffer,
                             int* startIndex) override {
        int actualIndexCount = 0;
        return this->makeIndexSpaceAtLeast(indexCount,
                                           indexCount,
                                           buffer,
                                           startIndex,
                                           &actualIndexCount);
    }

    void* makeVertexSpaceAtLeast(size_t vertexSize, int minVertexCount, int fallbackVertexCount,
                                 sk_sp<const GrBuffer>* buffer, int* startVertex,
                                 int* actualVertexCount) override {
        if (!buffer || !startVertex || !actualVertexCount || vertexSize == 0 || minVertexCount <= 0) {
            return nullptr;
        }
        const int allocCount = std::max(minVertexCount, fallbackVertexCount);
        if (allocCount <= 0) {
            return nullptr;
        }

        auto gpuBuffer = make_dynamic_cpu_buffer(
                vertexSize * static_cast<size_t>(allocCount), GrGpuBufferType::kVertex);
        *buffer = gpuBuffer;
        *startVertex = 0;
        *actualVertexCount = allocCount;
        fLastVertexBuffer = std::move(gpuBuffer);
        fLastVertexStride = vertexSize;
        return fLastVertexBuffer->writableData();
    }

    uint16_t* makeIndexSpaceAtLeast(int minIndexCount,
                                    int fallbackIndexCount,
                                    sk_sp<const GrBuffer>* buffer,
                                    int* startIndex,
                                    int* actualIndexCount) override {
        if (!buffer || !startIndex || !actualIndexCount || minIndexCount <= 0) {
            return nullptr;
        }
        const int allocCount = std::max(minIndexCount, fallbackIndexCount);
        if (allocCount <= 0) {
            return nullptr;
        }

        auto gpuBuffer = make_dynamic_cpu_buffer(
                sizeof(uint16_t) * static_cast<size_t>(allocCount), GrGpuBufferType::kIndex);
        *buffer = gpuBuffer;
        *startIndex = 0;
        *actualIndexCount = allocCount;
        fLastIndexBuffer = std::move(gpuBuffer);
        return reinterpret_cast<uint16_t*>(fLastIndexBuffer->writableData());
    }

    GrDrawIndirectWriter makeDrawIndirectSpace(int drawCount,
                                               sk_sp<const GrBuffer>* buffer,
                                               size_t* offsetInBytes) override {
        if (!buffer || !offsetInBytes || drawCount <= 0) {
            return {};
        }
        auto gpuBuffer = make_dynamic_cpu_buffer(
                sizeof(GrDrawIndirectCommand) * static_cast<size_t>(drawCount),
                GrGpuBufferType::kDrawIndirect);
        *buffer = gpuBuffer;
        *offsetInBytes = 0;
        fLastIndirectBuffer = std::move(gpuBuffer);
        return GrDrawIndirectWriter(fLastIndirectBuffer->writableData());
    }

    GrDrawIndexedIndirectWriter makeDrawIndexedIndirectSpace(int drawCount,
                                                             sk_sp<const GrBuffer>* buffer,
                                                             size_t* offsetInBytes) override {
        if (!buffer || !offsetInBytes || drawCount <= 0) {
            return {};
        }
        auto gpuBuffer = make_dynamic_cpu_buffer(
                sizeof(GrDrawIndexedIndirectCommand) * static_cast<size_t>(drawCount),
                GrGpuBufferType::kDrawIndirect);
        *buffer = gpuBuffer;
        *offsetInBytes = 0;
        fLastIndexedIndirectBuffer = std::move(gpuBuffer);
        return GrDrawIndexedIndirectWriter(fLastIndexedIndirectBuffer->writableData());
    }

    void putBackIndices(int indices) override {
        if (!fLastIndexBuffer || indices <= 0) {
            return;
        }
        const size_t bytesToTrim = sizeof(uint16_t) * static_cast<size_t>(indices);
        trim_buffer_tail(fLastIndexBuffer.get(), bytesToTrim);
    }

    void putBackVertices(int vertices, size_t vertexStride) override {
        if (!fLastVertexBuffer || vertices <= 0 || vertexStride == 0) {
            return;
        }
        if (vertexStride != fLastVertexStride) {
            return;
        }
        const size_t bytesToTrim = static_cast<size_t>(vertices) * vertexStride;
        trim_buffer_tail(fLastVertexBuffer.get(), bytesToTrim);
    }

    void putBackIndirectDraws(int count) override {
        if (!fLastIndirectBuffer || count <= 0) {
            return;
        }
        const size_t bytesToTrim = sizeof(GrDrawIndirectCommand) * static_cast<size_t>(count);
        trim_buffer_tail(fLastIndirectBuffer.get(), bytesToTrim);
    }

    void putBackIndexedIndirectDraws(int count) override {
        if (!fLastIndexedIndirectBuffer || count <= 0) {
            return;
        }
        const size_t bytesToTrim = sizeof(GrDrawIndexedIndirectCommand) *
                                   static_cast<size_t>(count);
        trim_buffer_tail(fLastIndexedIndirectBuffer.get(), bytesToTrim);
    }

    GrResourceProvider* resourceProvider() const override {
        return const_cast<CpuResourceProvider*>(&fResourceProvider);
    }

    GrRenderTargetProxy* rtProxy() const override { return fWriteView.asRenderTargetProxy(); }
    const GrSurfaceProxyView& writeView() const override { return fWriteView; }
    const GrAppliedClip* appliedClip() const override { return &fAppliedClip; }
    GrAppliedClip detachAppliedClip() override {
        GrAppliedClip clip = std::move(fAppliedClip);
        fAppliedClip.~GrAppliedClip();
        new (&fAppliedClip) GrAppliedClip(GrAppliedClip::Disabled());
        return clip;
    }
    const GrDstProxyView& dstProxyView() const override { return fDstProxyView; }
    bool usesMSAASurface() const override { return false; }
    GrXferBarrierFlags renderPassBarriers() const override { return GrXferBarrierFlags::kNone; }
    GrLoadOp colorLoadOp() const override { return GrLoadOp::kLoad; }
    GrThreadSafeCache* threadSafeCache() const override { return nullptr; }
    SkArenaAlloc* allocator() override { return &fAllocator; }
    const GrCaps& caps() const override { return fCaps; }
    sktext::gpu::StrikeCache* strikeCache() const override { return nullptr; }
    GrAtlasManager* atlasManager() const override { return nullptr; }
#if !defined(SK_ENABLE_OPTIMIZE_SIZE)
    skgpu::ganesh::SmallPathAtlasMgr* smallPathAtlasManager() const override { return nullptr; }
#endif
    skia_private::TArray<GrSurfaceProxy*, true>* sampledProxyArray() override {
        return &fSampledProxies;
    }
    GrDeferredUploadTarget* deferredUploadTarget() override { return nullptr; }

private:
    mutable CpuResourceProvider fResourceProvider;
    SkArenaAlloc fAllocator;
    GrCaps fCaps;
    GrRenderTargetProxy fDummyProxy{1, 1, 1};
    GrSurfaceProxyView fWriteView;
    GrDstProxyView fDstProxyView;
    GrAppliedClip fAppliedClip;
    skia_private::TArray<GrSurfaceProxy*, true> fSampledProxies;
    sk_sp<CpuGpuBuffer> fLastVertexBuffer;
    sk_sp<CpuGpuBuffer> fLastIndexBuffer;
    sk_sp<CpuGpuBuffer> fLastIndirectBuffer;
    sk_sp<CpuGpuBuffer> fLastIndexedIndirectBuffer;
    size_t fLastVertexStride = 0;
};

static bool append_patch_chunk_to_output(const GrVertexChunk& chunk,
                                         uint32_t patchStrideBytes,
                                         const CpuGpuBuffer& cpuBuffer,
                                         PatchBufferData* outPatches,
                                         size_t* byteOffset,
                                         std::string* error);

static bool copy_cpu_buffer_bytes(const CpuGpuBuffer& cpuBuffer,
                                  std::vector<uint8_t>* outBytes,
                                  const char* nullOutputMessage,
                                  std::string* error);

static void append_fan_pass(std::vector<TessPassPlan>* outPasses,
                            TessPlanPassKind kind,
                            const CpuProgramInfo& programInfo,
                            const Mat3& shaderMatrix,
                            const char* debugName,
                            MeshData fanMesh,
                            std::vector<TessDrawCommand> fanDraws);

template <typename FixedCountType>
static int resolve_level_from_fixed_vertex_count(int maxVertexCount, bool wedgeMode) {
    for (int resolveLevel = 0; resolveLevel <= skgpu::tess::kMaxResolveLevel; ++resolveLevel) {
        const int expected = wedgeMode
                ? (skgpu::tess::NumCurveTrianglesAtResolveLevel(resolveLevel) + 1) * 3
                : skgpu::tess::NumCurveTrianglesAtResolveLevel(resolveLevel) * 3;
        if (expected == maxVertexCount) {
            return resolveLevel;
        }
    }
    return skgpu::tess::kMaxResolveLevel;
}

template <typename FixedCountType, typename TessellatorT>
static bool export_tessellator_buffers(const TessellatorT& tessellator,
                                       uint32_t preallocPatchCount,
                                       bool wedgeMode,
                                       PatchBufferData* outPatches,
                                       std::string* error) {
    if (!outPatches) {
        set_error(error, "outPatches is null");
        return false;
    }

    *outPatches = {};
    outPatches->patchStrideBytes = static_cast<uint32_t>(
            skgpu::tess::PatchStride(tessellator.patchAttribs()));
    outPatches->attribMask = static_cast<uint32_t>(tessellator.patchAttribs());
    outPatches->preallocPatchCount = preallocPatchCount;
    outPatches->maxFixedCountVertexCount = path_tessellator_max_vertex_count(tessellator);
    outPatches->requiredResolveLevel =
            resolve_level_from_fixed_vertex_count<FixedCountType>(
                    path_tessellator_max_vertex_count(tessellator),
                                                                  wedgeMode);

    size_t byteOffset = 0;
    for (const GrVertexChunk& chunk : path_tessellator_vertex_chunks(tessellator)) {
        const auto* cpuBuffer = dynamic_cast<const CpuGpuBuffer*>(chunk.fBuffer.get());
        if (!cpuBuffer) {
            set_error(error, "Captured vertex chunk buffer is not a CpuGpuBuffer.");
            return false;
        }
        if (!append_patch_chunk_to_output(chunk,
                                          outPatches->patchStrideBytes,
                                          *cpuBuffer,
                                          outPatches,
                                          &byteOffset,
                                          error)) {
            return false;
        }
    }

    const auto* fixedVertexBuffer = dynamic_cast<const CpuGpuBuffer*>(
            path_tessellator_fixed_vertex_buffer(tessellator).get());
    const auto* fixedIndexBuffer = dynamic_cast<const CpuGpuBuffer*>(
            path_tessellator_fixed_index_buffer(tessellator).get());
    if (!fixedVertexBuffer || !fixedIndexBuffer) {
        set_error(error, "Tessellator did not return CPU-backed fixed template buffers.");
        return false;
    }

    if (!copy_cpu_buffer_bytes(*fixedVertexBuffer,
                               &outPatches->fixedVertexBufferTemplate,
                               "fixed vertex template output is null",
                               error)) {
        return false;
    }
    if (!copy_cpu_buffer_bytes(*fixedIndexBuffer,
                               &outPatches->fixedIndexBufferTemplate,
                               "fixed index template output is null",
                               error)) {
        return false;
    }
    outPatches->fixedVertexStrideBytes = static_cast<uint32_t>(FixedCountType::VertexBufferStride());
    outPatches->fixedVertexCount = static_cast<uint32_t>(FixedCountType::VertexBufferVertexCount());
    outPatches->fixedIndexCount = static_cast<uint32_t>(fixedIndexBuffer->size() / sizeof(uint16_t));
    return true;
}

template <typename CaptureFn>
static void capture_draw_commands(CpuMeshDrawTarget* target,
                                  CaptureFn&& captureFn,
                                  std::vector<TessDrawCommand>* outCommands) {
    if (!outCommands) {
        return;
    }
    CpuOpFlushState flushState(target->resourceProvider(), target->caps());
    captureFn(&flushState);
    *outCommands = flushState.drawCommands();
}

static bool export_point_buffer_as_mesh(const sk_sp<const GrBuffer>& buffer,
                                        int baseVertex,
                                        int vertexCount,
                                        MeshData* outMesh,
                                        std::string* error) {
    if (!outMesh) {
        set_error(error, "outMesh is null");
        return false;
    }
    outMesh->vertices.clear();
    outMesh->indices.clear();
    if (!buffer || vertexCount <= 0) {
        return true;
    }

    const auto* cpuBuffer = dynamic_cast<const CpuGpuBuffer*>(buffer.get());
    if (!cpuBuffer) {
        set_error(error, "Expected CpuGpuBuffer-backed point vertex buffer.");
        return false;
    }
    const size_t byteOffset = static_cast<size_t>(std::max(baseVertex, 0)) * sizeof(SkPoint);
    const size_t byteSize = static_cast<size_t>(std::max(vertexCount, 0)) * sizeof(SkPoint);
    if (byteOffset + byteSize > cpuBuffer->size()) {
        set_error(error, "Point vertex buffer slice exceeds backing buffer size.");
        return false;
    }

    const auto* points = reinterpret_cast<const SkPoint*>(cpuBuffer->data() + byteOffset);
    outMesh->vertices.reserve(static_cast<size_t>(vertexCount));
    outMesh->indices.reserve(static_cast<size_t>(vertexCount));
    for (int i = 0; i < vertexCount; ++i) {
        outMesh->vertices.push_back({points[i].x(), points[i].y()});
        outMesh->indices.push_back(static_cast<uint32_t>(i));
    }
    return true;
}

static bool export_instance_buffer(const sk_sp<const GrBuffer>& buffer,
                                   uint32_t strideBytes,
                                   int baseInstance,
                                   int instanceCount,
                                   TessInstanceBufferData* outInstances,
                                   std::string* error) {
    if (!outInstances) {
        set_error(error, "outInstances is null");
        return false;
    }
    outInstances->data.clear();
    outInstances->strideBytes = strideBytes;
    outInstances->instanceCount = 0;
    if (!buffer || instanceCount <= 0 || strideBytes == 0) {
        return true;
    }

    const auto* cpuBuffer = dynamic_cast<const CpuGpuBuffer*>(buffer.get());
    if (!cpuBuffer) {
        set_error(error, "Expected CpuGpuBuffer-backed instance buffer.");
        return false;
    }
    const size_t byteOffset = static_cast<size_t>(std::max(baseInstance, 0)) * strideBytes;
    const size_t byteSize = static_cast<size_t>(std::max(instanceCount, 0)) * strideBytes;
    if (byteOffset + byteSize > cpuBuffer->size()) {
        set_error(error, "Instance buffer slice exceeds backing buffer size.");
        return false;
    }
    outInstances->data.resize(byteSize);
    std::memcpy(outInstances->data.data(), cpuBuffer->data() + byteOffset, byteSize);
    outInstances->instanceCount = static_cast<uint32_t>(instanceCount);
    return true;
}

static void breadcrumbs_to_triangles(const GrInnerFanTriangulator::BreadcrumbTriangleList& breadcrumbs,
                                     std::vector<Triangle>* outTriangles) {
    if (!outTriangles) {
        return;
    }
    outTriangles->clear();
    outTriangles->reserve(static_cast<size_t>(std::max(breadcrumbs.count(), 0)));
    for (const auto* tri = breadcrumbs.head(); tri; tri = tri->fNext) {
        outTriangles->push_back({
                {tri->fPts[0].x(), tri->fPts[0].y()},
                {tri->fPts[1].x(), tri->fPts[1].y()},
                {tri->fPts[2].x(), tri->fPts[2].y()}});
    }
}

static uint32_t curve_patch_attrib_mask(bool infinitySupport) {
    return infinitySupport
            ? 0u
            : static_cast<uint32_t>(skgpu::tess::PatchAttribs::kExplicitCurveType);
}

SKGPU_DECLARE_STATIC_UNIQUE_KEY(gCaptureUnitQuadBufferKey);
SKGPU_DECLARE_STATIC_UNIQUE_KEY(gCaptureHullVertexBufferKey);

static void clear_fixed_templates(PatchBufferData* patch) {
    if (!patch) {
        return;
    }
    patch->fixedVertexBufferTemplate.clear();
    patch->fixedIndexBufferTemplate.clear();
    patch->fixedVertexStrideBytes = 0;
    patch->fixedVertexCount = 0;
    patch->fixedIndexCount = 0;
}

static bool set_fixed_vertex_template(PatchBufferData* patch,
                                      const void* bytes,
                                      size_t byteSize,
                                      uint32_t strideBytes,
                                      std::string* error) {
    if (!patch) {
        set_error(error, "patch is null");
        return false;
    }
    if ((byteSize > 0 && !bytes) || (byteSize > 0 && strideBytes == 0)) {
        set_error(error, "invalid fixed vertex template input");
        return false;
    }
    if (strideBytes > 0 && (byteSize % strideBytes) != 0) {
        set_error(error, "fixed vertex template size is not divisible by stride");
        return false;
    }

    clear_fixed_templates(patch);
    patch->fixedVertexBufferTemplate.resize(byteSize);
    if (byteSize > 0) {
        std::memcpy(patch->fixedVertexBufferTemplate.data(), bytes, byteSize);
    }
    patch->fixedVertexStrideBytes = strideBytes;
    patch->fixedVertexCount = (strideBytes > 0)
            ? static_cast<uint32_t>(byteSize / strideBytes)
            : 0;
    return true;
}

static void initialize_hull_patch_buffer_from_curve_pass(const PatchBufferData& curvePatch,
                                                         const std::vector<TessDrawCommand>& hullDraws,
                                                         PatchBufferData* outPatch) {
    if (!outPatch) {
        return;
    }
    *outPatch = curvePatch;
    clear_fixed_templates(outPatch);
    outPatch->requiredResolveLevel = 0;
    outPatch->preallocPatchCount = outPatch->patchCount;
    outPatch->maxFixedCountVertexCount = 0;
    for (const TessDrawCommand& draw : hullDraws) {
        outPatch->maxFixedCountVertexCount =
                std::max(outPatch->maxFixedCountVertexCount,
                         static_cast<int>(draw.elementCount));
    }
}

static bool append_patch_chunk_to_output(const GrVertexChunk& chunk,
                                         uint32_t patchStrideBytes,
                                         const CpuGpuBuffer& cpuBuffer,
                                         PatchBufferData* outPatches,
                                         size_t* byteOffset,
                                         std::string* error) {
    if (!outPatches || !byteOffset) {
        set_error(error, "Patch output pointers are null.");
        return false;
    }

    const size_t chunkBytes = static_cast<size_t>(chunk.fCount) * patchStrideBytes;
    if (chunkBytes > cpuBuffer.size()) {
        set_error(error, "Captured vertex chunk byte size exceeds backing buffer size.");
        return false;
    }

    PatchBufferData::Chunk outChunk;
    outChunk.basePatch = static_cast<uint32_t>(chunk.fBase);
    outChunk.patchCount = static_cast<uint32_t>(chunk.fCount);
    outChunk.byteOffset = static_cast<uint32_t>(*byteOffset);
    outChunk.byteSize = static_cast<uint32_t>(chunkBytes);
    outPatches->chunks.push_back(outChunk);

    const size_t oldSize = outPatches->data.size();
    outPatches->data.resize(oldSize + chunkBytes);
    std::memcpy(outPatches->data.data() + oldSize, cpuBuffer.data(), chunkBytes);

    *byteOffset += chunkBytes;
    outPatches->patchCount += static_cast<uint32_t>(chunk.fCount);
    return true;
}

static bool copy_cpu_buffer_bytes(const CpuGpuBuffer& cpuBuffer,
                                  std::vector<uint8_t>* outBytes,
                                  const char* nullOutputMessage,
                                  std::string* error) {
    if (!outBytes) {
        set_error(error, nullOutputMessage);
        return false;
    }
    outBytes->resize(cpuBuffer.size());
    if (!outBytes->empty()) {
        std::memcpy(outBytes->data(), cpuBuffer.data(), cpuBuffer.size());
    }
    return true;
}

static void append_fan_pass(std::vector<TessPassPlan>* outPasses,
                            TessPlanPassKind kind,
                            const CpuProgramInfo& programInfo,
                            const Mat3& shaderMatrix,
                            const char* debugName,
                            MeshData fanMesh,
                            std::vector<TessDrawCommand> fanDraws) {
    if (!outPasses) {
        return;
    }
    TessPassPlan fanPass;
    fanPass.kind = kind;
    fanPass.programInfo = programInfo.toPlanInfo();
    fanPass.shaderMatrix = shaderMatrix;
    fanPass.debugName = debugName;
    fanPass.triangleMesh = std::move(fanMesh);
    fanPass.drawCommands = std::move(fanDraws);
    outPasses->push_back(std::move(fanPass));
}

}  // namespace

bool CapturePathCurveTessellatorPrepareOriginalSkia(const SkPath& path,
                                                    const SkMatrix& shaderMatrix,
                                                    const SkMatrix& pathMatrix,
                                                    bool infinitySupport,
                                                    const std::vector<Triangle>& extraTriangles,
                                                    PatchBufferData* outPatches,
                                                    std::vector<TessDrawCommand>* outStencilDraws,
                                                    std::vector<TessDrawCommand>* outHullDraws,
                                                    std::string* error) {
    CpuMeshDrawTarget target(make_caps(infinitySupport, true));
    skgpu::ganesh::PathCurveTessellator tessellator(infinitySupport);

    SkArenaAlloc arena(1024 * 16);
    GrInnerFanTriangulator::BreadcrumbTriangleList breadcrumbs;
    for (const Triangle& tri : extraTriangles) {
        breadcrumbs.append(&arena,
                           SkPoint::Make(tri.p0.x, tri.p0.y),
                           SkPoint::Make(tri.p1.x, tri.p1.y),
                           SkPoint::Make(tri.p2.x, tri.p2.y),
                           1);
    }

    skgpu::ganesh::PathTessellator::PathDrawList drawList(pathMatrix,
                                                          path,
                                                          SK_PMColor4fTRANSPARENT);
    const uint32_t preallocPatchCount =
            static_cast<uint32_t>(skgpu::tess::FixedCountCurves::PreallocCount(path.countVerbs()) +
                                  static_cast<int>(extraTriangles.size()));
    tessellator.prepareWithTriangles(&target,
                                     shaderMatrix,
                                     &breadcrumbs,
                                     drawList,
                                     path.countVerbs());
    capture_draw_commands(&target,
                          [&tessellator](GrOpFlushState* flushState) {
                              tessellator.draw(flushState);
                          },
                          outStencilDraws);
    capture_draw_commands(&target,
                          [&tessellator](GrOpFlushState* flushState) {
                              tessellator.drawHullInstances(flushState, nullptr);
                          },
                          outHullDraws);
    return export_tessellator_buffers<skgpu::tess::FixedCountCurves>(tessellator,
                                                                     preallocPatchCount,
                                                                     false,
                                                                     outPatches,
                                                                     error);
}

bool CapturePathWedgeTessellatorPrepareOriginalSkia(const SkPath& path,
                                                    const SkMatrix& shaderMatrix,
                                                    const SkMatrix& pathMatrix,
                                                    bool infinitySupport,
                                                    PatchBufferData* outPatches,
                                                    std::vector<TessDrawCommand>* outStencilDraws,
                                                    std::string* error) {
    CpuMeshDrawTarget target(make_caps(infinitySupport, true));
    skgpu::ganesh::PathWedgeTessellator tessellator(infinitySupport);
    skgpu::ganesh::PathTessellator::PathDrawList drawList(pathMatrix,
                                                          path,
                                                          SK_PMColor4fTRANSPARENT);
    const uint32_t preallocPatchCount =
            static_cast<uint32_t>(skgpu::tess::FixedCountWedges::PreallocCount(path.countVerbs()));
    tessellator.prepare(&target, shaderMatrix, drawList, path.countVerbs());
    capture_draw_commands(&target,
                          [&tessellator](GrOpFlushState* flushState) {
                              tessellator.draw(flushState);
                          },
                          outStencilDraws);
    return export_tessellator_buffers<skgpu::tess::FixedCountWedges>(tessellator,
                                                                     preallocPatchCount,
                                                                     true,
                                                                     outPatches,
                                                                     error);
}

bool CapturePreparedPathStencilCoverOpOriginalSkia(const SkPath& path,
                                                   const PatchPrepareOptions& prepareOptions,
                                                   TessCapturePlan* outPlan,
                                                   std::string* error) {
    if (!outPlan) {
        set_error(error, "outPlan is null");
        return false;
    }
    *outPlan = {};
    outPlan->opKind = TessPlanOpKind::kPathStencilCoverOp;
    outPlan->usedOriginalSkiaCore = true;
    outPlan->complete = true;

    const SkMatrix viewMatrix = to_sk_matrix(prepareOptions.viewMatrix);
    SkRect drawBounds = viewMatrix.mapRect(path.getBounds());
    if (path.isInverseFillType()) {
        if (!prepareOptions.hasClipConservativeBounds) {
            set_error(error, "inverse fill requires clipConservativeBounds for draw bounds");
            return false;
        }
        drawBounds = to_sk_rect(prepareOptions.clipConservativeBounds);
    }
    const uint32_t curvePatchAttribMask = curve_patch_attrib_mask(prepareOptions.infinitySupport);
    const uint32_t wedgePatchAttribMask =
            static_cast<uint32_t>(skgpu::tess::PatchAttribs::kFanPoint) | curvePatchAttribMask;
    const CpuPathStencilCoverOpPrePrepare preprepare =
            CpuPrePreparePathStencilCoverOp(path,
                                            drawBounds,
                                            prepareOptions,
                                            curvePatchAttribMask,
                                            wedgePatchAttribMask);

    CpuMeshDrawTarget target(make_caps(prepareOptions.infinitySupport, prepareOptions.vertexIDSupport));

    if (preprepare.hasStencilFanProgram) {
        sk_sp<const GrBuffer> fanBuffer;
        int fanBaseVertex = 0;
        int fanVertexCount = 0;
        GrEagerDynamicVertexAllocator vertexAlloc(&target, &fanBuffer, &fanBaseVertex);
        const int maxTrianglesInFans = std::max(path.countVerbs() - 2, 0);
        if (skgpu::VertexWriter triangleVertexWriter =
                    vertexAlloc.lockWriter(sizeof(SkPoint), maxTrianglesInFans * 3)) {
            int fanTriangleCount = 0;
            skgpu::tess::AffineMatrix m(viewMatrix);
            for (skgpu::tess::PathMiddleOutFanIter it(path); !it.done();) {
                for (auto [p0, p1, p2] : it.nextStack()) {
                    triangleVertexWriter << m.map2Points(p0, p1) << m.mapPoint(p2);
                    ++fanTriangleCount;
                }
            }
            fanVertexCount = fanTriangleCount * 3;
            vertexAlloc.unlock(fanVertexCount);
        }

        if (fanVertexCount > 0) {
            TessPassPlan fanPass;
            fanPass.kind = TessPlanPassKind::kStencilFanTriangles;
            fanPass.programInfo = preprepare.stencilFanProgram.toPlanInfo();
            fanPass.shaderMatrix = Mat3{};
            fanPass.debugName = "PathStencilCoverOp.stencilFan";
            if (!export_point_buffer_as_mesh(fanBuffer,
                                             fanBaseVertex,
                                             fanVertexCount,
                                             &fanPass.triangleMesh,
                                             error)) {
                return false;
            }
            CpuOpFlushState fanFlush(target.resourceProvider(), target.caps());
            fanFlush.bindBuffers(nullptr, nullptr, fanBuffer);
            fanFlush.draw(fanVertexCount, fanBaseVertex);
            fanPass.drawCommands = fanFlush.drawCommands();
        outPlan->passes.push_back(std::move(fanPass));
    }

        TessPassPlan curvePass;
        curvePass.kind = TessPlanPassKind::kStencilCurvePatches;
        curvePass.programInfo = preprepare.stencilPathProgram.toPlanInfo();
        curvePass.shaderMatrix = Mat3{};
        curvePass.debugName = "PathStencilCoverOp.stencilCurves";
        if (!CapturePathCurveTessellatorPrepareOriginalSkia(path,
                                                            SkMatrix::I(),
                                                            viewMatrix,
                                                            prepareOptions.infinitySupport,
                                                            {},
                                                            &curvePass.patchBuffer,
                                                            &curvePass.drawCommands,
                                                            nullptr,
                                                            error)) {
            return false;
        }
        outPlan->passes.push_back(std::move(curvePass));
    } else if (preprepare.hasStencilPathProgram) {
        TessPassPlan wedgePass;
        wedgePass.kind = TessPlanPassKind::kStencilWedgePatches;
        wedgePass.programInfo = preprepare.stencilPathProgram.toPlanInfo();
        wedgePass.shaderMatrix = Mat3{};
        wedgePass.debugName = "PathStencilCoverOp.stencilWedges";
        if (!CapturePathWedgeTessellatorPrepareOriginalSkia(path,
                                                            SkMatrix::I(),
                                                            viewMatrix,
                                                            prepareOptions.infinitySupport,
                                                            &wedgePass.patchBuffer,
                                                            &wedgePass.drawCommands,
                                                            error)) {
            return false;
        }
        outPlan->passes.push_back(std::move(wedgePass));
    }

    if (!preprepare.hasCoverBBoxProgram) {
        return true;
    }

    constexpr uint32_t kBBoxInstanceStride = sizeof(float) * 10;
    sk_sp<const GrBuffer> bboxBuffer;
    int bboxBaseInstance = 0;
    if (skgpu::VertexWriter vertexWriter =
                target.makeVertexWriter(kBBoxInstanceStride, 1, &bboxBuffer, &bboxBaseInstance)) {
        SkRect bounds = path.getBounds();
        if (path.isInverseFillType()) {
            if (!prepareOptions.hasClipConservativeBounds) {
                set_error(error, "inverse fill requires clipConservativeBounds for bbox instance");
                return false;
            }
            const SkRect deviceBounds = to_sk_rect(prepareOptions.clipConservativeBounds);
            SkMatrix inv;
            if (viewMatrix.invert(&inv)) {
                bounds = inv.mapRect(deviceBounds);
            }
        }
        vertexWriter << viewMatrix.getScaleX()
                     << viewMatrix.getSkewY()
                     << viewMatrix.getSkewX()
                     << viewMatrix.getScaleY()
                     << viewMatrix.getTranslateX()
                     << viewMatrix.getTranslateY()
                     << bounds.left()
                     << bounds.top()
                     << bounds.right()
                     << bounds.bottom();
    } else {
        set_error(error, "Could not allocate bounding box instance buffer.");
        return false;
    }

    TessPassPlan bboxPass;
    bboxPass.kind = TessPlanPassKind::kCoverBoundingBoxes;
    bboxPass.programInfo = preprepare.coverBBoxProgram.toPlanInfo();
    bboxPass.shaderMatrix = Mat3{};
    bboxPass.debugName = "PathStencilCoverOp.coverBoundingBoxes";
    if (!export_instance_buffer(bboxBuffer,
                                kBBoxInstanceStride,
                                bboxBaseInstance,
                                1,
                                &bboxPass.instanceBuffer,
                                error)) {
        return false;
    }

    sk_sp<const GrBuffer> bboxVertexBufferIfNoIDSupport;
    if (!prepareOptions.vertexIDSupport) {
        constexpr static SkPoint kUnitQuad[4] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        SKGPU_DEFINE_STATIC_UNIQUE_KEY(gCaptureUnitQuadBufferKey);
        bboxVertexBufferIfNoIDSupport = target.resourceProvider()->findOrMakeStaticBuffer(
                GrGpuBufferType::kVertex,
                sizeof(kUnitQuad),
                kUnitQuad,
                gCaptureUnitQuadBufferKey);
        if (!set_fixed_vertex_template(&bboxPass.patchBuffer,
                                       kUnitQuad,
                                       sizeof(kUnitQuad),
                                       sizeof(SkPoint),
                                       error)) {
            return false;
        }
    }
    CpuOpFlushState bboxFlush(target.resourceProvider(), target.caps());
    bboxFlush.bindBuffers(nullptr, bboxBuffer, bboxVertexBufferIfNoIDSupport);
    bboxFlush.drawInstanced(1, bboxBaseInstance, 4, 0);
    bboxPass.drawCommands = bboxFlush.drawCommands();
    outPlan->passes.push_back(std::move(bboxPass));

    return true;
}

bool CapturePreparedPathInnerTriangulateOpOriginalSkia(const SkPath& path,
                                                       const PatchPrepareOptions& prepareOptions,
                                                       TessCapturePlan* outPlan,
                                                       std::string* error) {
    if (!outPlan) {
        set_error(error, "outPlan is null");
        return false;
    }
    *outPlan = {};
    outPlan->opKind = TessPlanOpKind::kPathInnerTriangulateOp;
    outPlan->usedOriginalSkiaCore = true;
    outPlan->complete = true;

    const SkMatrix viewMatrix = to_sk_matrix(prepareOptions.viewMatrix);
    const Mat3 shaderMatrix = prepareOptions.viewMatrix;
    CpuMeshDrawTarget target(make_caps(prepareOptions.infinitySupport, prepareOptions.vertexIDSupport));
    SkArenaAlloc alloc(GrTriangulator::kArenaDefaultChunkSize);
    GrInnerFanTriangulator triangulator(path, &alloc);
    bool isLinear = true;
    GrInnerFanTriangulator::BreadcrumbTriangleList fanBreadcrumbs;
    GrTriangulator::Poly* fanPolys = triangulator.pathToPolys(&fanBreadcrumbs, &isLinear);
    const uint32_t curvePatchAttribMask = curve_patch_attrib_mask(prepareOptions.infinitySupport);
    const CpuPathInnerTriangulateOpPrePrepare preprepare =
            CpuPrePreparePathInnerTriangulateOp(path,
                                                isLinear,
                                                fanPolys != nullptr,
                                                prepareOptions,
                                                curvePatchAttribMask);

    sk_sp<const GrBuffer> fanBuffer;
    int baseFanVertex = 0;
    int fanVertexCount = 0;
    if (fanPolys) {
        GrEagerDynamicVertexAllocator vertexAlloc(&target, &fanBuffer, &baseFanVertex);
        fanVertexCount = triangulator.polysToTriangles(fanPolys, &vertexAlloc, &fanBreadcrumbs);
    }

    std::vector<TessPassPlan> fanPasses;
    if (fanVertexCount > 0 && (preprepare.hasStencilFanProgram ||
                               preprepare.hasFillFanProgram ||
                               preprepare.hasSecondaryStencilFanProgram)) {
        MeshData fanMesh;
        if (!export_point_buffer_as_mesh(fanBuffer,
                                         baseFanVertex,
                                         fanVertexCount,
                                         &fanMesh,
                                         error)) {
            return false;
        }
        CpuOpFlushState fanFlush(target.resourceProvider(), target.caps());
        fanFlush.bindBuffers(nullptr, nullptr, fanBuffer);
        fanFlush.draw(fanVertexCount, baseFanVertex);
        const std::vector<TessDrawCommand> fanDraws = fanFlush.drawCommands();

        if (preprepare.hasStencilFanProgram) {
            append_fan_pass(&fanPasses,
                            TessPlanPassKind::kStencilFanTriangles,
                            preprepare.stencilFanProgram,
                            shaderMatrix,
                            "PathInnerTriangulateOp.stencilFan",
                            fanMesh,
                            fanDraws);
        }
        if (preprepare.hasFillFanProgram) {
            append_fan_pass(&fanPasses,
                            TessPlanPassKind::kFillFanTriangles,
                            preprepare.fillFanProgram,
                            shaderMatrix,
                            isLinear ? "PathInnerTriangulateOp.fillLinearFan"
                                     : "PathInnerTriangulateOp.fillInnerFan",
                            fanMesh,
                            fanDraws);
        }
        if (preprepare.hasSecondaryStencilFanProgram) {
            append_fan_pass(&fanPasses,
                            TessPlanPassKind::kStencilFanTriangles,
                            preprepare.secondaryStencilFanProgram,
                            shaderMatrix,
                            "PathInnerTriangulateOp.stencilFanAfterFill",
                            std::move(fanMesh),
                            fanDraws);
        }
    }

    if (preprepare.hasStencilCurvesProgram) {
        std::vector<Triangle> breadcrumbs;
        breadcrumbs_to_triangles(fanBreadcrumbs, &breadcrumbs);

        TessPassPlan curvePass;
        curvePass.kind = TessPlanPassKind::kStencilCurvePatches;
        curvePass.programInfo = preprepare.stencilCurvesProgram.toPlanInfo();
        curvePass.shaderMatrix = shaderMatrix;
        curvePass.debugName = "PathInnerTriangulateOp.stencilCurves";
        std::vector<TessDrawCommand> hullDraws;
        if (!CapturePathCurveTessellatorPrepareOriginalSkia(path,
                                                            viewMatrix,
                                                            SkMatrix::I(),
                                                            prepareOptions.infinitySupport,
                                                            breadcrumbs,
                                                            &curvePass.patchBuffer,
                                                            &curvePass.drawCommands,
                                                            &hullDraws,
                                                            error)) {
            return false;
        }
        if (!prepareOptions.vertexIDSupport) {
            for (TessDrawCommand& command : hullDraws) {
                command.boundVertexBufferBytes = sizeof(float) * 4;
            }
        }
        outPlan->passes.push_back(curvePass);

        for (auto& fanPass : fanPasses) {
            outPlan->passes.push_back(std::move(fanPass));
        }
        fanPasses.clear();

        if (preprepare.hasCoverHullsProgram) {
            TessPassPlan hullPass;
            hullPass.kind = TessPlanPassKind::kCoverHulls;
            hullPass.programInfo = preprepare.coverHullsProgram.toPlanInfo();
            hullPass.shaderMatrix = shaderMatrix;
            hullPass.debugName = "PathInnerTriangulateOp.coverHulls";
            initialize_hull_patch_buffer_from_curve_pass(curvePass.patchBuffer,
                                                         hullDraws,
                                                         &hullPass.patchBuffer);
            hullPass.drawCommands = std::move(hullDraws);

            if (!prepareOptions.vertexIDSupport) {
                constexpr static float kStripOrderIDs[4] = {0, 1, 3, 2};
                SKGPU_DEFINE_STATIC_UNIQUE_KEY(gCaptureHullVertexBufferKey);
                [[maybe_unused]] auto hullVertexBufferIfNoIDSupport =
                        target.resourceProvider()->findOrMakeStaticBuffer(
                                GrGpuBufferType::kVertex,
                                sizeof(kStripOrderIDs),
                                kStripOrderIDs,
                                gCaptureHullVertexBufferKey);
                if (!set_fixed_vertex_template(&hullPass.patchBuffer,
                                               kStripOrderIDs,
                                               sizeof(kStripOrderIDs),
                                               sizeof(float),
                                               error)) {
                    return false;
                }
            }
            outPlan->passes.push_back(std::move(hullPass));
        }
    } else {
        for (auto& fanPass : fanPasses) {
            outPlan->passes.push_back(std::move(fanPass));
        }
    }

    return true;
}

static SkPaint::Join to_sk_join(StrokeJoin join) {
    switch (join) {
        case StrokeJoin::kBevel: return SkPaint::kBevel_Join;
        case StrokeJoin::kRound: return SkPaint::kRound_Join;
        case StrokeJoin::kMiter: default: return SkPaint::kMiter_Join;
    }
}

static SkPaint::Cap to_sk_cap(StrokeCap cap) {
    switch (cap) {
        case StrokeCap::kSquare: return SkPaint::kSquare_Cap;
        case StrokeCap::kRound: return SkPaint::kRound_Cap;
        case StrokeCap::kButt: default: return SkPaint::kButt_Cap;
    }
}

bool CaptureStrokeTessellatorPrepareOriginalSkia(const SkPath& path,
                                                 const StrokeOptions& strokeOptions,
                                                 const SkMatrix& shaderMatrix,
                                                 const SkMatrix& pathMatrix,
                                                 bool infinitySupport,
                                                 bool vertexIDSupport,
                                                 PatchBufferData* outPatches,
                                                 std::vector<TessDrawCommand>* outDraws,
                                                 std::string* error) {
    if (!outPatches || !outDraws) {
        set_error(error, "Output pointers are null.");
        return false;
    }
    *outPatches = {};
    outDraws->clear();

    if (path.isEmpty()) {
        return true;
    }

    // Upstream stroke tessellation only keys off the shader/view matrix. There is no separate
    // per-path matrix input on StrokeTessellator::prepare().
    (void)pathMatrix;

    SkStrokeRec strokeRec(SkStrokeRec::kHairline_InitStyle);
    if (strokeOptions.width > 0.0f) {
        strokeRec.setStrokeStyle(strokeOptions.width);
        strokeRec.setStrokeParams(to_sk_cap(strokeOptions.cap),
                                  to_sk_join(strokeOptions.join),
                                  strokeOptions.miterLimit);
    }

    const SkPMColor4f color = {1, 1, 1, 1};
    skgpu::ganesh::StrokeTessellator::PathStrokeList pathStrokeList(path, strokeRec, color);

    PatchAttribs attribs = PatchAttribs::kNone;
    if (!infinitySupport) {
        attribs |= PatchAttribs::kExplicitCurveType;
    }
    // StrokeTessellator will automatically add kJoinControlPoint inside its constructor.

    skgpu::ganesh::StrokeTessellator tessellator(attribs);

    CpuMeshDrawTarget target(make_caps(infinitySupport, vertexIDSupport));
    tessellator.prepare(&target, shaderMatrix, &pathStrokeList, path.countVerbs());

    const PatchAttribs resolvedAttribs = stroke_tessellator_patch_attribs(tessellator);
    outPatches->patchStrideBytes = static_cast<uint32_t>(skgpu::tess::PatchStride(resolvedAttribs));
    outPatches->attribMask = static_cast<uint32_t>(resolvedAttribs);
    outPatches->preallocPatchCount =
            static_cast<uint32_t>(skgpu::tess::FixedCountStrokes::PreallocCount(path.countVerbs()));

    size_t byteOffset = 0;
    for (const GrVertexChunk& chunk : stroke_tessellator_vertex_chunks(tessellator)) {
        const auto* cpuBuffer = dynamic_cast<const CpuGpuBuffer*>(chunk.fBuffer.get());
        if (!cpuBuffer) {
            set_error(error, "Captured vertex chunk buffer is not a CpuGpuBuffer.");
            return false;
        }
        if (!append_patch_chunk_to_output(chunk,
                                          outPatches->patchStrideBytes,
                                          *cpuBuffer,
                                          outPatches,
                                          &byteOffset,
                                          error)) {
            return false;
        }
    }

    outPatches->maxFixedCountVertexCount = stroke_tessellator_max_vertex_count(tessellator);

    // Unlike curve/wedge tessellation, upstream StrokeTessellator never uses an index template.
    outPatches->fixedIndexBufferTemplate.clear();
    outPatches->fixedIndexCount = 0;

    if (stroke_tessellator_fixed_vertex_buffer(tessellator)) {
        if (const auto* cpuFixedBuf =
                    dynamic_cast<const CpuGpuBuffer*>(
                            stroke_tessellator_fixed_vertex_buffer(tessellator).get())) {
            outPatches->fixedVertexStrideBytes = sizeof(float);
            if (!copy_cpu_buffer_bytes(*cpuFixedBuf,
                                       &outPatches->fixedVertexBufferTemplate,
                                       "fixed vertex template output is null",
                                       error)) {
                return false;
            }
            outPatches->fixedVertexCount = static_cast<uint32_t>(
                    outPatches->fixedVertexBufferTemplate.size() / sizeof(float));
        } else {
            set_error(error, "Stroke fallback vertex buffer is not CpuGpuBuffer-backed.");
            return false;
        }
    }

    capture_draw_commands(&target,
                          [&tessellator](GrOpFlushState* flushState) {
                              tessellator.draw(flushState);
                          },
                          outDraws);

    return true;
}

bool CaptureStrokeTessellatorPrepareOriginalSkia(const SkPath& path,
                                                 const StrokeOptions& strokeOptions,
                                                 const SkMatrix& shaderMatrix,
                                                 const SkMatrix& pathMatrix,
                                                 bool infinitySupport,
                                                 PatchBufferData* outPatches,
                                                 std::vector<TessDrawCommand>* outDraws,
                                                 std::string* error) {
    return CaptureStrokeTessellatorPrepareOriginalSkia(path,
                                                       strokeOptions,
                                                       shaderMatrix,
                                                       pathMatrix,
                                                       infinitySupport,
                                                       true,
                                                       outPatches,
                                                       outDraws,
                                                       error);
}

}  // namespace skia_port
