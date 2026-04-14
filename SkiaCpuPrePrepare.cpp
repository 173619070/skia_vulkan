#include "SkiaCpuPrePrepare.h"

#include <ported_skia/include/core/SkMatrix.h>
#include <ported_skia/include/core/SkStrokeRec.h>
#include <ported_skia/src/base/SkArenaAlloc.h>
#include <ported_skia/src/core/SkColorData.h>
#include <ported_skia/src/gpu/ganesh/GrGeometryProcessor.h>
#include <ported_skia/src/gpu/ganesh/GrShaderCaps.h>
#include <ported_skia/src/gpu/ganesh/tessellate/GrPathTessellationShader.h>
#include <ported_skia/src/gpu/ganesh/tessellate/GrStrokeTessellationShader.h>
#include <ported_skia/src/gpu/tessellate/Tessellation.h>

namespace skia_port {

namespace {

static TessStencilSettingsKind fill_rule_to_stencil_settings(const SkPath& path) {
    return path.getFillType() == SkPathFillType::kWinding
            ? TessStencilSettingsKind::kFillOrIncrDecr
            : TessStencilSettingsKind::kFillOrInvert;
}

static TessStencilSettingsKind cover_bounding_box_stencil_settings(const SkPath& path) {
    return SkPathFillType_IsInverse(path.getFillType())
            ? TessStencilSettingsKind::kTestAndResetInverse
            : TessStencilSettingsKind::kTestAndReset;
}

static TessStencilSettingsKind inner_triangulate_fill_fan_stencil_settings(
        const SkPath& path,
        const CpuAppliedClip& appliedClip,
        bool forceRedbookStencilPass,
        bool isLinear) {
    if (forceRedbookStencilPass) {
        return TessStencilSettingsKind::kTestAndReset;
    }
    if (isLinear) {
        return TessStencilSettingsKind::kUnused;
    }
    if (!appliedClip.hasStencilClip()) {
        return fill_rule_to_stencil_settings(path);
    }
    return TessStencilSettingsKind::kFillIfZeroAndInClip;
}

static TessStencilSettingsKind inner_triangulate_secondary_stencil_fan_settings(
        const SkPath& path) {
    return path.getFillType() == SkPathFillType::kWinding
            ? TessStencilSettingsKind::kIncrDecrStencilIfNonzero
            : TessStencilSettingsKind::kInvertStencilIfNonzero;
}

static bool use_curve_tessellator_for_path(const SkPath& path, const SkRect& drawBounds) {
    return path.countVerbs() > 50 && drawBounds.height() * drawBounds.width() > 256.0f * 256.0f;
}

static TessLayoutKind to_layout_kind(GrGeometryProcessor::LayoutKind kind) {
    switch (kind) {
        case GrGeometryProcessor::LayoutKind::kSimpleTriangle:
            return TessLayoutKind::kSimpleTriangle;
        case GrGeometryProcessor::LayoutKind::kMiddleOut:
            return TessLayoutKind::kMiddleOut;
        case GrGeometryProcessor::LayoutKind::kBoundingBox:
            return TessLayoutKind::kBoundingBox;
        case GrGeometryProcessor::LayoutKind::kHull:
            return TessLayoutKind::kHull;
        case GrGeometryProcessor::LayoutKind::kStroke:
            return TessLayoutKind::kStroke;
        case GrGeometryProcessor::LayoutKind::kUnknown:
        default:
            return TessLayoutKind::kUnknown;
    }
}

static TessStencilFaceInfo make_stencil_face(TessStencilTestKind test,
                                             TessStencilOpKind passOp,
                                             TessStencilOpKind failOp,
                                             uint16_t ref = 0,
                                             uint16_t testMask = 0xffff,
                                             uint16_t writeMask = 0xffff) {
    TessStencilFaceInfo face;
    face.ref = ref;
    face.test = test;
    face.testMask = testMask;
    face.passOp = passOp;
    face.failOp = failOp;
    face.writeMask = writeMask;
    return face;
}

static CpuPrimitiveProcessorLayout layout_from_gp(const GrGeometryProcessor& gp,
                                                  bool usesVertexID) {
    return CpuPrimitiveProcessorLayout(to_layout_kind(gp.layoutKind()),
                                       static_cast<uint32_t>(gp.vertexStride()),
                                       static_cast<uint32_t>(gp.instanceStride()),
                                       gp.vertexAttributeCount(),
                                       gp.instanceAttributeCount(),
                                       gp.hasVertexAttributes(),
                                       gp.hasInstanceAttributes(),
                                       usesVertexID);
}

static GrShaderCaps to_gr_caps(const CpuShaderCaps& caps) {
    GrShaderCaps out;
    out.setInfinitySupport(caps.infinitySupport());
    out.setVertexIDSupport(caps.vertexIDSupport());
    return out;
}

static CpuPrimitiveProcessorLayout make_simple_triangle_layout(const CpuShaderCaps& caps) {
    const GrShaderCaps shaderCaps = to_gr_caps(caps);
    SkArenaAlloc arena(256);
    auto* shader = GrPathTessellationShader::MakeSimpleTriangleShader(
            &arena, SkMatrix::I(), SK_PMColor4fTRANSPARENT);
    return layout_from_gp(*shader, shaderCaps.vertexIDSupport());
}

static CpuPrimitiveProcessorLayout make_middle_out_layout(uint32_t patchAttribMask,
                                                          const CpuShaderCaps& caps) {
    const GrShaderCaps shaderCaps = to_gr_caps(caps);
    SkArenaAlloc arena(256);
    auto* shader = GrPathTessellationShader::Make(
            shaderCaps,
            &arena,
            SkMatrix::I(),
            SK_PMColor4fTRANSPARENT,
            static_cast<skgpu::tess::PatchAttribs>(patchAttribMask));
    return layout_from_gp(*shader, shaderCaps.vertexIDSupport());
}

static CpuPrimitiveProcessorLayout make_bounding_box_layout(bool usesVertexID) {
    return CpuPrimitiveProcessorLayout(TessLayoutKind::kBoundingBox,
                                       usesVertexID ? 0u : static_cast<uint32_t>(sizeof(float) * 2),
                                       static_cast<uint32_t>(sizeof(float) * 10),
                                       usesVertexID ? 0u : 1u,
                                       3,
                                       !usesVertexID,
                                       true,
                                       usesVertexID);
}

static CpuPrimitiveProcessorLayout make_hull_layout(uint32_t patchAttribMask, bool usesVertexID) {
    using PatchAttribs = skgpu::tess::PatchAttribs;
    const bool hasExplicitCurveType =
            static_cast<PatchAttribs>(patchAttribMask) & PatchAttribs::kExplicitCurveType;
    return CpuPrimitiveProcessorLayout(TessLayoutKind::kHull,
                                       usesVertexID ? 0u : static_cast<uint32_t>(sizeof(float)),
                                       static_cast<uint32_t>(sizeof(SkPoint) * 4 +
                                                             (hasExplicitCurveType
                                                                     ? sizeof(float)
                                                                     : 0)),
                                       usesVertexID ? 0u : 1u,
                                       hasExplicitCurveType ? 3u : 2u,
                                       !usesVertexID,
                                       true,
                                       usesVertexID);
}

static CpuPrimitiveProcessorLayout make_stroke_layout(uint32_t patchAttribMask,
                                                      const CpuShaderCaps& caps) {
    const GrShaderCaps shaderCaps = to_gr_caps(caps);
    SkArenaAlloc arena(256);
    SkStrokeRec strokeRec(SkStrokeRec::kHairline_InitStyle);
    auto* shader = arena.make<GrStrokeTessellationShader>(
            shaderCaps,
            static_cast<skgpu::tess::PatchAttribs>(patchAttribMask),
            SkMatrix::I(),
            strokeRec,
            SK_PMColor4fTRANSPARENT);
    return layout_from_gp(*shader, shaderCaps.vertexIDSupport());
}

struct CpuPrePrepareContext {
    explicit CpuPrePrepareContext(const PatchPrepareOptions& options)
            : caps(options)
            , appliedClip(options.hasStencilClip ? CpuAppliedClip(true)
                                                 : CpuAppliedClip::Disabled())
            , stencilPipeline(appliedClip.hardClip().hasStencilClip(),
                              options.wireframe,
                              options.stencilOnly)
            , fillPipeline(appliedClip.hardClip().hasStencilClip(),
                           options.wireframe,
                           options.stencilOnly) {}

    CpuCaps caps;
    CpuAppliedClip appliedClip;
    CpuPipeline stencilPipeline;
    CpuPipeline fillPipeline;
};

static const CpuShaderCaps& shader_caps(const CpuPrePrepareContext& ctx) {
    return *ctx.caps.shaderCaps();
}

static bool vertex_id_support(const CpuPrePrepareContext& ctx) {
    return ctx.caps.shaderCaps()->vertexIDSupport();
}

static CpuProgramInfo make_simple_triangle_program(TessProgramKind kind,
                                                   TessStencilSettingsKind stencilKind,
                                                   const CpuPrePrepareContext& ctx,
                                                   const CpuPipeline& pipeline,
                                                   bool usesVertexID) {
    return CpuProgramInfo(kind,
                          TessPrimitiveType::kTriangles,
                          CpuStencilSettings(stencilKind),
                          make_simple_triangle_layout(shader_caps(ctx)),
                          pipeline,
                          0,
                          usesVertexID);
}

static CpuProgramInfo make_middle_out_program(TessProgramKind kind,
                                              TessStencilSettingsKind stencilKind,
                                              const CpuPrePrepareContext& ctx,
                                              const CpuPipeline& pipeline,
                                              uint32_t patchAttribMask) {
    return CpuProgramInfo(kind,
                          TessPrimitiveType::kPatches,
                          CpuStencilSettings(stencilKind),
                          make_middle_out_layout(patchAttribMask, shader_caps(ctx)),
                          pipeline,
                          patchAttribMask,
                          vertex_id_support(ctx));
}

static CpuProgramInfo make_bounding_box_program(TessStencilSettingsKind stencilKind,
                                                const CpuPrePrepareContext& ctx) {
    return CpuProgramInfo(TessProgramKind::kCoverBoundingBox,
                          TessPrimitiveType::kTriangleStrip,
                          CpuStencilSettings(stencilKind),
                          make_bounding_box_layout(vertex_id_support(ctx)),
                          ctx.fillPipeline,
                          0,
                          vertex_id_support(ctx));
}

static CpuProgramInfo make_hull_program(TessProgramKind kind,
                                        TessStencilSettingsKind stencilKind,
                                        const CpuPrePrepareContext& ctx,
                                        const CpuPipeline& pipeline,
                                        uint32_t patchAttribMask) {
    return CpuProgramInfo(kind,
                          TessPrimitiveType::kTriangleStrip,
                          CpuStencilSettings(stencilKind),
                          make_hull_layout(patchAttribMask, vertex_id_support(ctx)),
                          pipeline,
                          patchAttribMask,
                          vertex_id_support(ctx));
}

static CpuProgramInfo make_stroke_program(TessProgramKind kind,
                                          TessStencilSettingsKind stencilKind,
                                          const CpuPrePrepareContext& ctx,
                                          const CpuPipeline& pipeline,
                                          uint32_t patchAttribMask) {
    return CpuProgramInfo(kind,
                          TessPrimitiveType::kTriangleStrip,
                          CpuStencilSettings(stencilKind),
                          make_stroke_layout(patchAttribMask, shader_caps(ctx)),
                          pipeline,
                          patchAttribMask,
                          vertex_id_support(ctx));
}

static void configure_path_stencil_cover_programs(const SkPath& path,
                                                  const CpuPrePrepareContext& ctx,
                                                  bool useCurveTessellator,
                                                  bool hasStencilPathProgram,
                                                  uint32_t curvePatchAttribMask,
                                                  uint32_t wedgePatchAttribMask,
                                                  CpuPathStencilCoverOpPrePrepare* out) {
    if (!out) {
        return;
    }

    if (useCurveTessellator) {
        out->hasStencilFanProgram = true;
        out->stencilFanProgram =
                make_simple_triangle_program(TessProgramKind::kStencilFan,
                                             fill_rule_to_stencil_settings(path),
                                             ctx,
                                             ctx.stencilPipeline,
                                             vertex_id_support(ctx));
        out->stencilPathProgram = make_middle_out_program(TessProgramKind::kStencilPath,
                                                          fill_rule_to_stencil_settings(path),
                                                          ctx,
                                                          ctx.stencilPipeline,
                                                          curvePatchAttribMask);
    } else if (hasStencilPathProgram) {
        out->stencilPathProgram = make_middle_out_program(TessProgramKind::kStencilPath,
                                                          fill_rule_to_stencil_settings(path),
                                                          ctx,
                                                          ctx.stencilPipeline,
                                                          wedgePatchAttribMask);
    }
}

static void assign_inner_triangulate_fill_fan_program(
        const SkPath& path,
        bool isLinear,
        const CpuPrePrepareContext& ctx,
        const CpuPathInnerTriangulateOpPrePrepare& config,
        CpuPathInnerTriangulateOpPrePrepare* out) {
    if (!out || !config.doFill) {
        return;
    }

    out->hasFillFanProgram = true;
    out->fillFanProgram = make_simple_triangle_program(
            TessProgramKind::kFanFill,
            inner_triangulate_fill_fan_stencil_settings(
                    path, ctx.appliedClip, config.forceRedbookStencilPass, isLinear),
            ctx,
            ctx.fillPipeline,
            true);
}

static void configure_inner_triangulate_fan_programs(
        const SkPath& path,
        bool isLinear,
        const CpuPrePrepareContext& ctx,
        CpuPathInnerTriangulateOpPrePrepare* out) {
    if (!out) {
        return;
    }

    if (out->forceRedbookStencilPass) {
        out->hasStencilFanProgram = true;
        out->stencilFanProgram =
                make_simple_triangle_program(TessProgramKind::kFanStencil,
                                             fill_rule_to_stencil_settings(path),
                                             ctx,
                                             ctx.stencilPipeline,
                                             true);
        assign_inner_triangulate_fill_fan_program(path, isLinear, ctx, *out, out);
        return;
    }

    assign_inner_triangulate_fill_fan_program(path, isLinear, ctx, *out, out);
    if (isLinear || !ctx.appliedClip.hasStencilClip()) {
        return;
    }

    out->hasSecondaryStencilFanProgram = true;
    out->secondaryStencilFanProgram =
            make_simple_triangle_program(TessProgramKind::kFanStencil,
                                         inner_triangulate_secondary_stencil_fan_settings(path),
                                         ctx,
                                         ctx.stencilPipeline,
                                         true);
}

static void initialize_inner_triangulate_state(const PatchPrepareOptions& options,
                                               bool isLinear,
                                               CpuPathInnerTriangulateOpPrePrepare* out) {
    if (!out) {
        return;
    }
    out->forceRedbookStencilPass = options.stencilOnly || options.wireframe;
    out->doFill = !options.stencilOnly;
    out->isLinear = isLinear;
    out->hasStencilCurvesProgram = !isLinear;
    out->hasCoverHullsProgram = out->doFill && !isLinear;
}

static void configure_inner_triangulate_curve_and_hull_programs(
        const SkPath& path,
        const CpuPrePrepareContext& ctx,
        uint32_t curvePatchAttribMask,
        CpuPathInnerTriangulateOpPrePrepare* out) {
    if (!out) {
        return;
    }

    if (out->hasStencilCurvesProgram) {
        out->stencilCurvesProgram = make_middle_out_program(TessProgramKind::kStencilCurves,
                                                            fill_rule_to_stencil_settings(path),
                                                            ctx,
                                                            ctx.stencilPipeline,
                                                            curvePatchAttribMask);
    }

    if (out->hasCoverHullsProgram) {
        out->coverHullsProgram = make_hull_program(TessProgramKind::kCoverHulls,
                                                   TessStencilSettingsKind::kTestAndReset,
                                                   ctx,
                                                   ctx.fillPipeline,
                                                   curvePatchAttribMask);
    }
}

static CpuPathTessellateOpPrePrepare make_path_tessellate_preprepare(
        const CpuPrePrepareContext& ctx,
        uint32_t patchAttribMask) {
    CpuPathTessellateOpPrePrepare out;
    out.tessellateProgram = make_middle_out_program(
            TessProgramKind::kStencilPath, // Represents GrPathTessellationShader
            TessStencilSettingsKind::kUnused,
            ctx,
            ctx.fillPipeline,
            patchAttribMask);
    return out;
}

static CpuStrokeTessellateOpPrePrepare make_stroke_tessellate_preprepare(
        const CpuPrePrepareContext& ctx,
        uint32_t patchAttribMask) {
    CpuStrokeTessellateOpPrePrepare out;
    // In this basic capture, we assume no stencil logic is needed for StrokeTessellateOp
    // unless explicitly specified. The Demo treats all draws as SrcOver without hit testing for now.
    out.needsStencil = false;
    out.fillProgram = make_stroke_program(
            TessProgramKind::kStencilPath, // Stub for GrStrokeTessellationShader
            TessStencilSettingsKind::kUnused,
            ctx,
            ctx.fillPipeline,
            patchAttribMask);
    return out;
}

}  // namespace

CpuShaderCaps::CpuShaderCaps(const PatchPrepareOptions& options)
        : fInfinitySupport(options.infinitySupport)
        , fVertexIDSupport(options.vertexIDSupport) {}

CpuCaps::CpuCaps(const PatchPrepareOptions& options) : fShaderCaps(options) {}

CpuPipeline::CpuPipeline(bool hasStencilClip, bool wireframe, bool stencilOnly)
        : fHasStencilClip(hasStencilClip)
        , fWireframe(wireframe)
        , fStencilOnly(stencilOnly) {}

CpuStencilSettings::CpuStencilSettings(TessStencilSettingsKind kind) : fKind(kind) {}

TessStencilSettingsInfo CpuStencilSettings::toPlanInfo() const {
    TessStencilSettingsInfo info;
    info.kind = fKind;
    switch (fKind) {
        case TessStencilSettingsKind::kUnused:
            break;
        case TessStencilSettingsKind::kStencilPath:
            info.usesStencil = true;
            info.writesStencil = true;
            info.usesClipBit = true;
            info.twoSided = true;
            info.front = make_stencil_face(TessStencilTestKind::kAlwaysIfInClip,
                                           TessStencilOpKind::kIncWrap,
                                           TessStencilOpKind::kKeep);
            info.back = make_stencil_face(TessStencilTestKind::kAlwaysIfInClip,
                                          TessStencilOpKind::kDecWrap,
                                          TessStencilOpKind::kKeep);
            break;
        case TessStencilSettingsKind::kTestAndReset:
            info.usesStencil = true;
            info.testsStencil = true;
            info.writesStencil = true;
            info.resetsStencil = true;
            info.front = make_stencil_face(TessStencilTestKind::kNotEqual,
                                           TessStencilOpKind::kZero,
                                           TessStencilOpKind::kKeep);
            info.back = info.front;
            break;
        case TessStencilSettingsKind::kTestAndResetInverse:
            info.usesStencil = true;
            info.testsStencil = true;
            info.writesStencil = true;
            info.resetsStencil = true;
            info.front = make_stencil_face(TessStencilTestKind::kEqual,
                                           TessStencilOpKind::kKeep,
                                           TessStencilOpKind::kZero);
            info.back = info.front;
            break;
        case TessStencilSettingsKind::kFillOrIncrDecr:
            info.usesStencil = true;
            info.testsStencil = true;
            info.writesStencil = true;
            info.twoSided = true;
            info.front = make_stencil_face(TessStencilTestKind::kEqual,
                                           TessStencilOpKind::kKeep,
                                           TessStencilOpKind::kIncWrap);
            info.back = make_stencil_face(TessStencilTestKind::kEqual,
                                          TessStencilOpKind::kKeep,
                                          TessStencilOpKind::kDecWrap);
            break;
        case TessStencilSettingsKind::kFillOrInvert:
            info.usesStencil = true;
            info.testsStencil = true;
            info.writesStencil = true;
            info.resetsStencil = true;
            info.front = make_stencil_face(TessStencilTestKind::kEqual,
                                           TessStencilOpKind::kKeep,
                                           TessStencilOpKind::kZero);
            info.back = info.front;
            break;
        case TessStencilSettingsKind::kFillIfZeroAndInClip:
            info.usesStencil = true;
            info.testsStencil = true;
            info.usesClipBit = true;
            info.front = make_stencil_face(TessStencilTestKind::kEqualIfInClip,
                                           TessStencilOpKind::kKeep,
                                           TessStencilOpKind::kKeep);
            info.back = info.front;
            break;
        case TessStencilSettingsKind::kIncrDecrStencilIfNonzero:
            info.usesStencil = true;
            info.testsStencil = true;
            info.writesStencil = true;
            info.twoSided = true;
            info.front = make_stencil_face(TessStencilTestKind::kNotEqual,
                                           TessStencilOpKind::kIncWrap,
                                           TessStencilOpKind::kKeep);
            info.back = make_stencil_face(TessStencilTestKind::kNotEqual,
                                          TessStencilOpKind::kDecWrap,
                                          TessStencilOpKind::kKeep);
            break;
        case TessStencilSettingsKind::kInvertStencilIfNonzero:
            info.usesStencil = true;
            info.testsStencil = true;
            info.writesStencil = true;
            info.resetsStencil = true;
            info.front = make_stencil_face(TessStencilTestKind::kNotEqual,
                                           TessStencilOpKind::kZero,
                                           TessStencilOpKind::kKeep);
            info.back = info.front;
            break;
        case TessStencilSettingsKind::kUnknown:
            break;
    }
    return info;
}

CpuPrimitiveProcessorLayout::CpuPrimitiveProcessorLayout(TessLayoutKind kind,
                                                         uint32_t vertexStrideBytes,
                                                         uint32_t instanceStrideBytes,
                                                         uint32_t vertexAttributeCount,
                                                         uint32_t instanceAttributeCount,
                                                         bool hasVertexAttributes,
                                                         bool hasInstanceAttributes,
                                                         bool usesVertexID)
        : fKind(kind)
        , fVertexStrideBytes(vertexStrideBytes)
        , fInstanceStrideBytes(instanceStrideBytes)
        , fVertexAttributeCount(vertexAttributeCount)
        , fInstanceAttributeCount(instanceAttributeCount)
        , fHasVertexAttributes(hasVertexAttributes)
        , fHasInstanceAttributes(hasInstanceAttributes)
        , fUsesVertexID(usesVertexID) {}

TessPrimitiveProcessorLayout CpuPrimitiveProcessorLayout::toPlanInfo() const {
    TessPrimitiveProcessorLayout info;
    info.kind = fKind;
    info.vertexStrideBytes = fVertexStrideBytes;
    info.instanceStrideBytes = fInstanceStrideBytes;
    info.vertexAttributeCount = fVertexAttributeCount;
    info.instanceAttributeCount = fInstanceAttributeCount;
    info.hasVertexAttributes = fHasVertexAttributes;
    info.hasInstanceAttributes = fHasInstanceAttributes;
    info.usesVertexID = fUsesVertexID;
    return info;
}

CpuProgramInfo::CpuProgramInfo(TessProgramKind kind,
                               TessPrimitiveType primitiveType,
                               CpuStencilSettings stencilSettings,
                               CpuPrimitiveProcessorLayout layout,
                               CpuPipeline pipeline,
                               uint32_t patchAttribMask,
                               bool usesVertexID)
        : fKind(kind)
        , fPrimitiveType(primitiveType)
        , fStencilSettings(stencilSettings)
        , fLayout(std::move(layout))
        , fPipeline(std::move(pipeline))
        , fPatchAttribMask(patchAttribMask)
        , fUsesVertexID(usesVertexID) {}

TessProgramInfo CpuProgramInfo::toPlanInfo() const {
    TessProgramInfo info;
    info.kind = fKind;
    info.primitiveType = fPrimitiveType;
    info.stencilSettings = fStencilSettings.toPlanInfo();
    info.layout = fLayout.toPlanInfo();
    info.patchAttribMask = fPatchAttribMask;
    info.usesVertexID = fUsesVertexID;
    info.hasStencilClip = fPipeline.hasStencilClip();
    info.wireframe = fPipeline.isWireframe();
    info.stencilOnly = fPipeline.isStencilOnly();
    return info;
}

CpuPathStencilCoverOpPrePrepare CpuPrePreparePathStencilCoverOp(const SkPath& path,
                                                                const SkRect& drawBounds,
                                                                const PatchPrepareOptions& options,
                                                                uint32_t curvePatchAttribMask,
                                                                uint32_t wedgePatchAttribMask) {
    CpuPathStencilCoverOpPrePrepare out;
    CpuPrePrepareContext ctx(options);

    out.useCurveTessellator = use_curve_tessellator_for_path(path, drawBounds);
    out.hasStencilPathProgram = path.countVerbs() > 0;
    out.hasCoverBBoxProgram = !options.stencilOnly;

    configure_path_stencil_cover_programs(path,
                                          ctx,
                                          out.useCurveTessellator,
                                          out.hasStencilPathProgram,
                                          curvePatchAttribMask,
                                          wedgePatchAttribMask,
                                          &out);

    if (out.hasCoverBBoxProgram) {
        out.coverBBoxProgram =
                make_bounding_box_program(cover_bounding_box_stencil_settings(path), ctx);
    }

    return out;
}

CpuPathInnerTriangulateOpPrePrepare CpuPrePreparePathInnerTriangulateOp(
        const SkPath& path,
        bool isLinear,
        bool hasFanPolys,
        const PatchPrepareOptions& options,
        uint32_t curvePatchAttribMask) {
    CpuPathInnerTriangulateOpPrePrepare out;
    CpuPrePrepareContext ctx(options);

    initialize_inner_triangulate_state(options, isLinear, &out);
    configure_inner_triangulate_curve_and_hull_programs(path, ctx, curvePatchAttribMask, &out);

    if (!hasFanPolys) {
        return out;
    }

    configure_inner_triangulate_fan_programs(path, isLinear, ctx, &out);

    return out;
}

CpuPathTessellateOpPrePrepare CpuPrePreparePathTessellateOp(const SkPath&,
                                                            const PatchPrepareOptions& options,
                                                            uint32_t patchAttribMask) {
    CpuPrePrepareContext ctx(options);
    return make_path_tessellate_preprepare(ctx, patchAttribMask);
}

CpuStrokeTessellateOpPrePrepare CpuPrePrepareStrokeTessellateOp(const SkPath&,
                                                                const PatchPrepareOptions& options,
                                                                uint32_t patchAttribMask) {
    CpuPrePrepareContext ctx(options);
    return make_stroke_tessellate_preprepare(ctx, patchAttribMask);
}

}  // namespace skia_port
