#pragma once

#include "SkiaPathMeshPort.h"
#include "SkiaCpuGpuBuffers.h"
#include <ported_skia/include/core/SkRefCnt.h>
#include <ported_skia/include/private/base/SkTArray.h>
#include <ported_skia/src/gpu/ganesh/GrCaps.h>
#include <ported_skia/src/gpu/ganesh/GrOpFlushState.h>
#include <ported_skia/src/gpu/ganesh/GrDstProxyView.h>
#include <ported_skia/src/gpu/ganesh/GrSurfaceProxyView.h>

#include <memory>
#include <vector>

class GrAtlasManager;
class GrBuffer;
class GrGpuBuffer;
class GrOp;
class GrResourceProvider;
class GrSurfaceProxy;
class GrThreadSafeCache;
class GrRenderTargetProxy;
class GrDeferredUploadTarget;

namespace skgpu::ganesh {
class SmallPathAtlasMgr;
}

namespace sktext::gpu {
class StrikeCache;
}

namespace skia_port {

class CpuOpFlushState final : public GrOpFlushState {
public:
    CpuOpFlushState(GrResourceProvider* resourceProvider, const GrCaps& caps);

    void bindPipeline(const GrProgramInfo& programInfo, const SkRect& drawBounds) override;
    void setScissorRect(const SkIRect& scissorRect) override;
    void bindTextures(const GrGeometryProcessor&,
                      const GrSurfaceProxy* const[],
                      const GrPipeline&) override {}

    void* makeVertexSpace(size_t vertexSize, int vertexCount, sk_sp<const GrBuffer>*,
                          int* startVertex) override;
    uint16_t* makeIndexSpace(int indexCount, sk_sp<const GrBuffer>*, int* startIndex) override;
    void* makeVertexSpaceAtLeast(size_t vertexSize, int minVertexCount, int fallbackVertexCount,
                                 sk_sp<const GrBuffer>*, int* startVertex,
                                 int* actualVertexCount) override;
    uint16_t* makeIndexSpaceAtLeast(int minIndexCount, int fallbackIndexCount,
                                    sk_sp<const GrBuffer>*, int* startIndex,
                                    int* actualIndexCount) override;
    void putBackVertices(int vertices, size_t vertexStride) override;
    void putBackIndices(int indices) override;
    GrDrawIndirectWriter makeDrawIndirectSpace(int drawCount, sk_sp<const GrBuffer>* buffer,
                                               size_t* offsetInBytes) override;
    GrDrawIndexedIndirectWriter makeDrawIndexedIndirectSpace(int drawCount,
                                                             sk_sp<const GrBuffer>* buffer,
                                                             size_t* offsetInBytes) override;
    void putBackIndirectDraws(int count) override;
    void putBackIndexedIndirectDraws(int count) override;

    SkArenaAlloc* allocator() override;

    void recordDraw(const GrGeometryProcessor*,
                    const GrSimpleMesh[],
                    int,
                    const GrSurfaceProxy* const[],
                    GrPrimitiveType) override;
    void bindBuffers(sk_sp<const GrBuffer> indexBuffer,
                     sk_sp<const GrBuffer> instanceBuffer,
                     sk_sp<const GrBuffer> vertexBuffer) override;
    void draw(int vertexCount, int baseVertex) override;
    void drawIndexedInstanced(int indexCount,
                              int baseIndex,
                              int instanceCount,
                              int baseInstance,
                              int baseVertex) override;
    void drawInstanced(int instanceCount,
                       int baseInstance,
                       int vertexCount,
                       int baseVertex) override;
    GrResourceProvider* resourceProvider() const override;
    const GrCaps& caps() const override;
    GrAppliedClip detachAppliedClip() override;
    GrThreadSafeCache* threadSafeCache() const override;
    sktext::gpu::StrikeCache* strikeCache() const override;
    GrAtlasManager* atlasManager() const override;
#if !defined(SK_ENABLE_OPTIMIZE_SIZE)
    skgpu::ganesh::SmallPathAtlasMgr* smallPathAtlasManager() const override;
#endif
    GrDeferredUploadTarget* deferredUploadTarget() override;

    void setCurrentOp(GrOp* op);
    void clearCurrentOp();
    void setWriteViewSize(int width, int height, int sampleCount = 1);
    void setUsesMSAASurface(bool usesMSAA);
    void setAppliedClip(GrAppliedClip clip);

    const std::vector<TessDrawCommand>& drawCommands() const { return fDrawCommands; }

    struct CapturedDraw {
        const GrProgramInfo* program = nullptr;
        TessDrawCommand command;
        sk_sp<const GrBuffer> indexBuffer;
        sk_sp<const GrBuffer> instanceBuffer;
        sk_sp<const GrBuffer> vertexBuffer;
    };

    const std::vector<CapturedDraw>& capturedDraws() const { return fCapturedDraws; }

private:
    void refreshCurrentOpArgs();
    TessDrawCommand makeBaseCommand() const;

    GrResourceProvider* fResourceProvider = nullptr;
    GrCaps fCaps;
    GrRenderTargetProxy fDummyProxy{1, 1, 1};
    GrSurfaceProxyView fWriteView;
    GrDstProxyView fDstProxyView;
    GrAppliedClip fAppliedClip{SkISize::Make(1 << 29, 1 << 29)};
    bool fUsesMSAA = false;
    GrXferBarrierFlags fBarrierFlags = GrXferBarrierFlags::kNone;
    GrLoadOp fLoadOp = GrLoadOp::kLoad;
    SkArenaAlloc fAllocator{1024 * 16};

    sk_sp<CpuGpuBuffer> fLastVertexBuffer;
    size_t fLastVertexStride = 0;
    sk_sp<CpuGpuBuffer> fLastIndexBuffer;
    sk_sp<CpuGpuBuffer> fLastIndirectBuffer;
    sk_sp<CpuGpuBuffer> fLastIndexedIndirectBuffer;

    const GrProgramInfo* fActiveProgram = nullptr;

    sk_sp<const GrBuffer> fBoundIndexBuffer;
    sk_sp<const GrBuffer> fBoundInstanceBuffer;
    sk_sp<const GrBuffer> fBoundVertexBuffer;
    SkIRect fScissorRect = SkIRect::MakeEmpty();
    std::vector<TessDrawCommand> fDrawCommands;
    std::vector<CapturedDraw> fCapturedDraws;
    GrOp* fCurrentOp = nullptr;
    std::unique_ptr<GrOpFlushState::OpArgs> fCurrentOpArgs;
};

}  // namespace skia_port
