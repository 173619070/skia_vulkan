#pragma once

#include "SkiaPathMeshPort.h"
#include <ported_skia/include/core/SkPath.h>
#include <ported_skia/include/core/SkRect.h>

namespace skia_port {

// Phase 6 keeps CpuPrePrepare* as low-level debug/assert helpers only.
// Formal tessellation plans now come from runtime capture in SkiaTessCapture.cpp.

class CpuShaderCaps {
public:
    explicit CpuShaderCaps(const PatchPrepareOptions& options);

    bool infinitySupport() const { return fInfinitySupport; }
    bool vertexIDSupport() const { return fVertexIDSupport; }

private:
    bool fInfinitySupport = false;
    bool fVertexIDSupport = true;
};

class CpuCaps {
public:
    explicit CpuCaps(const PatchPrepareOptions& options);

    const CpuShaderCaps* shaderCaps() const { return &fShaderCaps; }

private:
    CpuShaderCaps fShaderCaps;
};

class CpuAppliedHardClip {
public:
    explicit CpuAppliedHardClip(bool hasStencilClip) : fHasStencilClip(hasStencilClip) {}

    bool hasStencilClip() const { return fHasStencilClip; }

private:
    bool fHasStencilClip = false;
};

class CpuAppliedClip {
public:
    explicit CpuAppliedClip(bool hasStencilClip) : fHardClip(hasStencilClip) {}

    static CpuAppliedClip Disabled() { return CpuAppliedClip(false); }

    const CpuAppliedHardClip& hardClip() const { return fHardClip; }
    bool hasStencilClip() const { return fHardClip.hasStencilClip(); }

private:
    CpuAppliedHardClip fHardClip;
};

class CpuPipeline {
public:
    CpuPipeline(bool hasStencilClip, bool wireframe, bool stencilOnly);

    bool hasStencilClip() const { return fHasStencilClip; }
    bool isWireframe() const { return fWireframe; }
    bool isStencilOnly() const { return fStencilOnly; }

private:
    bool fHasStencilClip = false;
    bool fWireframe = false;
    bool fStencilOnly = false;
};

class CpuStencilSettings {
public:
    CpuStencilSettings() = default;
    explicit CpuStencilSettings(TessStencilSettingsKind kind);

    TessStencilSettingsKind kind() const { return fKind; }
    TessStencilSettingsInfo toPlanInfo() const;

private:
    TessStencilSettingsKind fKind = TessStencilSettingsKind::kUnknown;
};

class CpuPrimitiveProcessorLayout {
public:
    CpuPrimitiveProcessorLayout() = default;
    CpuPrimitiveProcessorLayout(TessLayoutKind kind,
                                uint32_t vertexStrideBytes,
                                uint32_t instanceStrideBytes,
                                uint32_t vertexAttributeCount,
                                uint32_t instanceAttributeCount,
                                bool hasVertexAttributes,
                                bool hasInstanceAttributes,
                                bool usesVertexID);

    TessPrimitiveProcessorLayout toPlanInfo() const;

private:
    TessLayoutKind fKind = TessLayoutKind::kUnknown;
    uint32_t fVertexStrideBytes = 0;
    uint32_t fInstanceStrideBytes = 0;
    uint32_t fVertexAttributeCount = 0;
    uint32_t fInstanceAttributeCount = 0;
    bool fHasVertexAttributes = false;
    bool fHasInstanceAttributes = false;
    bool fUsesVertexID = true;
};

class CpuProgramInfo {
public:
    CpuProgramInfo() = default;
    CpuProgramInfo(TessProgramKind kind,
                   TessPrimitiveType primitiveType,
                   CpuStencilSettings stencilSettings,
                   CpuPrimitiveProcessorLayout layout,
                   CpuPipeline pipeline,
                   uint32_t patchAttribMask,
                   bool usesVertexID);

    const CpuPipeline& pipeline() const { return fPipeline; }
    TessProgramKind kind() const { return fKind; }
    TessPrimitiveType primitiveType() const { return fPrimitiveType; }
    const CpuStencilSettings& stencilSettings() const { return fStencilSettings; }
    const CpuPrimitiveProcessorLayout& layout() const { return fLayout; }
    uint32_t patchAttribMask() const { return fPatchAttribMask; }
    bool usesVertexID() const { return fUsesVertexID; }

    TessProgramInfo toPlanInfo() const;

private:
    TessProgramKind fKind = TessProgramKind::kUnknown;
    TessPrimitiveType fPrimitiveType = TessPrimitiveType::kUnknown;
    CpuStencilSettings fStencilSettings;
    CpuPrimitiveProcessorLayout fLayout;
    CpuPipeline fPipeline = CpuPipeline(false, false, false);
    uint32_t fPatchAttribMask = 0;
    bool fUsesVertexID = true;
};

struct CpuPathStencilCoverOpPrePrepare {
    bool useCurveTessellator = false;
    bool hasStencilFanProgram = false;
    bool hasStencilPathProgram = false;
    bool hasCoverBBoxProgram = false;
    CpuProgramInfo stencilFanProgram;
    CpuProgramInfo stencilPathProgram;
    CpuProgramInfo coverBBoxProgram;
};

struct CpuPathInnerTriangulateOpPrePrepare {
    bool forceRedbookStencilPass = false;
    bool doFill = true;
    bool isLinear = true;
    bool hasStencilCurvesProgram = false;
    bool hasStencilFanProgram = false;
    bool hasFillFanProgram = false;
    bool hasSecondaryStencilFanProgram = false;
    bool hasCoverHullsProgram = false;
    CpuProgramInfo stencilCurvesProgram;
    CpuProgramInfo stencilFanProgram;
    CpuProgramInfo fillFanProgram;
    CpuProgramInfo secondaryStencilFanProgram;
    CpuProgramInfo coverHullsProgram;
};

struct CpuPathTessellateOpPrePrepare {
    CpuProgramInfo tessellateProgram;
};

struct CpuStrokeTessellateOpPrePrepare {
    bool needsStencil = false;
    CpuProgramInfo stencilProgram;
    CpuProgramInfo fillProgram;
};

CpuPathStencilCoverOpPrePrepare CpuPrePreparePathStencilCoverOp(const SkPath& path,
                                                                const SkRect& drawBounds,
                                                                const PatchPrepareOptions& options,
                                                                uint32_t curvePatchAttribMask,
                                                                uint32_t wedgePatchAttribMask);

CpuPathInnerTriangulateOpPrePrepare CpuPrePreparePathInnerTriangulateOp(
        const SkPath& path,
        bool isLinear,
        bool hasFanPolys,
        const PatchPrepareOptions& options,
        uint32_t curvePatchAttribMask);

CpuPathTessellateOpPrePrepare CpuPrePreparePathTessellateOp(const SkPath& path,
                                                            const PatchPrepareOptions& options,
                                                            uint32_t patchAttribMask);

CpuStrokeTessellateOpPrePrepare CpuPrePrepareStrokeTessellateOp(const SkPath& path,
                                                                const PatchPrepareOptions& options,
                                                                uint32_t patchAttribMask);

}  // namespace skia_port
