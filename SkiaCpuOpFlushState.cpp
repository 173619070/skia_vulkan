#include "SkiaCpuOpFlushState.h"

#include <ported_skia/src/gpu/ganesh/GrBuffer.h>
#include <ported_skia/src/gpu/ganesh/GrGeometryProcessor.h>
#include <ported_skia/src/gpu/ganesh/ops/GrOp.h>
#include <ported_skia/src/gpu/ganesh/GrPipeline.h>
#include <ported_skia/src/gpu/ganesh/GrProgramInfo.h>
#include <ported_skia/src/gpu/ganesh/GrResourceProvider.h>

#include <algorithm>
#include <new>

namespace skia_port {

namespace {

static sk_sp<CpuGpuBuffer> make_dynamic_cpu_buffer(size_t byteSize, GrGpuBufferType type) {
    return sk_make_sp<CpuGpuBuffer>(byteSize, type, kDynamic_GrAccessPattern);
}

template <typename BufferT>
static void trim_buffer_tail(BufferT* buffer, size_t bytesToTrim) {
    if (!buffer) {
        return;
    }
    if (bytesToTrim > buffer->size()) {
        buffer->resize(0);
        return;
    }
    buffer->resize(buffer->size() - bytesToTrim);
}

static uint32_t buffer_size_bytes(const sk_sp<const GrBuffer>& buffer) {
    return buffer ? static_cast<uint32_t>(buffer->size()) : 0;
}

static void capture_bound_draw(const GrProgramInfo* activeProgram,
                               const TessDrawCommand& command,
                               const sk_sp<const GrBuffer>& boundIndexBuffer,
                               const sk_sp<const GrBuffer>& boundInstanceBuffer,
                               const sk_sp<const GrBuffer>& boundVertexBuffer,
                               std::vector<CpuOpFlushState::CapturedDraw>* outCapturedDraws) {
    if (!activeProgram || !outCapturedDraws) {
        return;
    }
    CpuOpFlushState::CapturedDraw draw;
    draw.program = activeProgram;
    draw.command = command;
    draw.indexBuffer = boundIndexBuffer;
    draw.instanceBuffer = boundInstanceBuffer;
    draw.vertexBuffer = boundVertexBuffer;
    outCapturedDraws->push_back(std::move(draw));
}

static void append_draw_command(const GrProgramInfo* activeProgram,
                                const TessDrawCommand& command,
                                const sk_sp<const GrBuffer>& boundIndexBuffer,
                                const sk_sp<const GrBuffer>& boundInstanceBuffer,
                                const sk_sp<const GrBuffer>& boundVertexBuffer,
                                std::vector<TessDrawCommand>* outDrawCommands,
                                std::vector<CpuOpFlushState::CapturedDraw>* outCapturedDraws) {
    if (outDrawCommands) {
        outDrawCommands->push_back(command);
    }
    capture_bound_draw(activeProgram,
                       command,
                       boundIndexBuffer,
                       boundInstanceBuffer,
                       boundVertexBuffer,
                       outCapturedDraws);
}

}  // namespace

CpuOpFlushState::CpuOpFlushState(GrResourceProvider* resourceProvider, const GrCaps& caps)
        : fResourceProvider(resourceProvider)
        , fCaps(caps)
        , fWriteView(&fDummyProxy) {}

void CpuOpFlushState::bindPipeline(const GrProgramInfo& programInfo,
                                   const SkRect&) {
    fActiveProgram = &programInfo;
}

void CpuOpFlushState::setScissorRect(const SkIRect& scissorRect) {
    fScissorRect = scissorRect;
}

void* CpuOpFlushState::makeVertexSpace(size_t vertexSize, int vertexCount,
                                       sk_sp<const GrBuffer>* buffer,
                                       int* startVertex) {
    int actualVertexCount = 0;
    return this->makeVertexSpaceAtLeast(vertexSize,
                                        vertexCount,
                                        vertexCount,
                                        buffer,
                                        startVertex,
                                        &actualVertexCount);
}

uint16_t* CpuOpFlushState::makeIndexSpace(int indexCount,
                                          sk_sp<const GrBuffer>* buffer,
                                          int* startIndex) {
    int actualIndexCount = 0;
    return this->makeIndexSpaceAtLeast(indexCount,
                                       indexCount,
                                       buffer,
                                       startIndex,
                                       &actualIndexCount);
}

void* CpuOpFlushState::makeVertexSpaceAtLeast(size_t vertexSize,
                                              int minVertexCount,
                                              int fallbackVertexCount,
                                              sk_sp<const GrBuffer>* buffer,
                                              int* startVertex,
                                              int* actualVertexCount) {
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

uint16_t* CpuOpFlushState::makeIndexSpaceAtLeast(int minIndexCount,
                                                 int fallbackIndexCount,
                                                 sk_sp<const GrBuffer>* buffer,
                                                 int* startIndex,
                                                 int* actualIndexCount) {
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

void CpuOpFlushState::putBackVertices(int vertices, size_t vertexStride) {
    if (!fLastVertexBuffer || vertices <= 0 || vertexStride == 0) {
        return;
    }
    if (vertexStride != fLastVertexStride) {
        return;
    }
    const size_t bytesToTrim = static_cast<size_t>(vertices) * vertexStride;
    trim_buffer_tail(fLastVertexBuffer.get(), bytesToTrim);
}

void CpuOpFlushState::putBackIndices(int indices) {
    if (!fLastIndexBuffer || indices <= 0) {
        return;
    }
    const size_t bytesToTrim = sizeof(uint16_t) * static_cast<size_t>(indices);
    trim_buffer_tail(fLastIndexBuffer.get(), bytesToTrim);
}

GrDrawIndirectWriter CpuOpFlushState::makeDrawIndirectSpace(int drawCount,
                                                            sk_sp<const GrBuffer>* buffer,
                                                            size_t* offsetInBytes) {
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

GrDrawIndexedIndirectWriter CpuOpFlushState::makeDrawIndexedIndirectSpace(
        int drawCount,
        sk_sp<const GrBuffer>* buffer,
        size_t* offsetInBytes) {
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

void CpuOpFlushState::putBackIndirectDraws(int count) {
    if (!fLastIndirectBuffer || count <= 0) {
        return;
    }
    const size_t bytesToTrim = sizeof(GrDrawIndirectCommand) * static_cast<size_t>(count);
    trim_buffer_tail(fLastIndirectBuffer.get(), bytesToTrim);
}

void CpuOpFlushState::putBackIndexedIndirectDraws(int count) {
    if (!fLastIndexedIndirectBuffer || count <= 0) {
        return;
    }
    const size_t bytesToTrim = sizeof(GrDrawIndexedIndirectCommand) * static_cast<size_t>(count);
    trim_buffer_tail(fLastIndexedIndirectBuffer.get(), bytesToTrim);
}

SkArenaAlloc* CpuOpFlushState::allocator() {
    return &fAllocator;
}

void CpuOpFlushState::recordDraw(const GrGeometryProcessor*,
                                 const GrSimpleMesh[],
                                 int,
                                 const GrSurfaceProxy* const[],
                                 GrPrimitiveType) {}

void CpuOpFlushState::bindBuffers(sk_sp<const GrBuffer> indexBuffer,
                                  sk_sp<const GrBuffer> instanceBuffer,
                                  sk_sp<const GrBuffer> vertexBuffer) {
    fBoundIndexBuffer = std::move(indexBuffer);
    fBoundInstanceBuffer = std::move(instanceBuffer);
    fBoundVertexBuffer = std::move(vertexBuffer);
}

void CpuOpFlushState::draw(int vertexCount, int baseVertex) {
    TessDrawCommand command = this->makeBaseCommand();
    command.kind = TessDrawCommandKind::kDraw;
    command.elementCount = static_cast<uint32_t>(vertexCount);
    // GrOpFlushState::draw() is a non-instanced draw, but Vulkan indirect draw commands still
    // require instanceCount=1 to execute. Keeping this at 0 causes fan/simple-triangle passes to
    // be captured yet replay as no-op draws.
    command.instanceCount = 1;
    command.baseVertex = static_cast<uint32_t>(baseVertex);
    append_draw_command(fActiveProgram,
                        command,
                        fBoundIndexBuffer,
                        fBoundInstanceBuffer,
                        fBoundVertexBuffer,
                        &fDrawCommands,
                        &fCapturedDraws);
}

void CpuOpFlushState::drawIndexedInstanced(int indexCount,
                                           int baseIndex,
                                           int instanceCount,
                                           int baseInstance,
                                           int baseVertex) {
    TessDrawCommand command = this->makeBaseCommand();
    command.kind = TessDrawCommandKind::kIndexedInstanced;
    command.elementCount = static_cast<uint32_t>(indexCount);
    command.baseIndex = static_cast<uint32_t>(baseIndex);
    command.instanceCount = static_cast<uint32_t>(instanceCount);
    command.baseInstance = static_cast<uint32_t>(baseInstance);
    command.baseVertex = static_cast<uint32_t>(baseVertex);
    append_draw_command(fActiveProgram,
                        command,
                        fBoundIndexBuffer,
                        fBoundInstanceBuffer,
                        fBoundVertexBuffer,
                        &fDrawCommands,
                        &fCapturedDraws);
}

void CpuOpFlushState::drawInstanced(int instanceCount,
                                    int baseInstance,
                                    int vertexCount,
                                    int baseVertex) {
    TessDrawCommand command = this->makeBaseCommand();
    command.kind = TessDrawCommandKind::kInstanced;
    command.elementCount = static_cast<uint32_t>(vertexCount);
    command.instanceCount = static_cast<uint32_t>(instanceCount);
    command.baseInstance = static_cast<uint32_t>(baseInstance);
    command.baseVertex = static_cast<uint32_t>(baseVertex);
    append_draw_command(fActiveProgram,
                        command,
                        fBoundIndexBuffer,
                        fBoundInstanceBuffer,
                        fBoundVertexBuffer,
                        &fDrawCommands,
                        &fCapturedDraws);
}

GrResourceProvider* CpuOpFlushState::resourceProvider() const {
    return fResourceProvider;
}

const GrCaps& CpuOpFlushState::caps() const {
    return fCaps;
}

GrAppliedClip CpuOpFlushState::detachAppliedClip() {
    GrAppliedClip clip = std::move(fAppliedClip);
    fAppliedClip.~GrAppliedClip();
    new (&fAppliedClip) GrAppliedClip(GrAppliedClip::Disabled());
    this->refreshCurrentOpArgs();
    return clip;
}

GrThreadSafeCache* CpuOpFlushState::threadSafeCache() const {
    return nullptr;
}

sktext::gpu::StrikeCache* CpuOpFlushState::strikeCache() const {
    return nullptr;
}

GrAtlasManager* CpuOpFlushState::atlasManager() const {
    return nullptr;
}

#if !defined(SK_ENABLE_OPTIMIZE_SIZE)
skgpu::ganesh::SmallPathAtlasMgr* CpuOpFlushState::smallPathAtlasManager() const {
    return nullptr;
}
#endif

GrDeferredUploadTarget* CpuOpFlushState::deferredUploadTarget() {
    return nullptr;
}

void CpuOpFlushState::setCurrentOp(GrOp* op) {
    fCurrentOp = op;
    this->refreshCurrentOpArgs();
}

void CpuOpFlushState::clearCurrentOp() {
    fCurrentOp = nullptr;
    fCurrentOpArgs.reset();
    this->setOpArgs(nullptr);
}

void CpuOpFlushState::setWriteViewSize(int width, int height, int sampleCount) {
    width = std::max(width, 1);
    height = std::max(height, 1);
    sampleCount = std::max(sampleCount, 1);
    fDummyProxy = GrRenderTargetProxy(width, height, sampleCount);
    fWriteView = GrSurfaceProxyView(&fDummyProxy);
    this->refreshCurrentOpArgs();
}

void CpuOpFlushState::setUsesMSAASurface(bool usesMSAA) {
    fUsesMSAA = usesMSAA;
    this->refreshCurrentOpArgs();
}

void CpuOpFlushState::setAppliedClip(GrAppliedClip clip) {
    fAppliedClip.~GrAppliedClip();
    new (&fAppliedClip) GrAppliedClip(std::move(clip));
    this->refreshCurrentOpArgs();
}

void CpuOpFlushState::refreshCurrentOpArgs() {
    if (!fCurrentOp) {
        fCurrentOpArgs.reset();
        this->setOpArgs(nullptr);
        return;
    }
    fCurrentOpArgs = std::make_unique<GrOpFlushState::OpArgs>(fCurrentOp,
                                                              fWriteView,
                                                              fUsesMSAA,
                                                              &fAppliedClip,
                                                              fDstProxyView,
                                                              fBarrierFlags,
                                                              fLoadOp);
    this->setOpArgs(fCurrentOpArgs.get());
}

TessDrawCommand CpuOpFlushState::makeBaseCommand() const {
    TessDrawCommand command;
    command.boundIndexBufferBytes = buffer_size_bytes(fBoundIndexBuffer);
    command.boundInstanceBufferBytes = buffer_size_bytes(fBoundInstanceBuffer);
    command.boundVertexBufferBytes = buffer_size_bytes(fBoundVertexBuffer);
    return command;
}

}  // namespace skia_port
