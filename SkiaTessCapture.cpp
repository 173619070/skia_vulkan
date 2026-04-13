#include "SkiaPathMeshPort.h"
#include "SkiaCpuGpuBuffers.h"
#include "SkiaCpuMeshDrawTarget.h"
#include "SkiaCpuOpFlushState.h"
#include "SkiaSimpleHardClip.h"

#include <ported_skia/include/core/SkMatrix.h>
#include <ported_skia/include/core/SkPath.h>
#include <ported_skia/include/core/SkPoint.h>
#include <ported_skia/include/core/SkRect.h>
#include <ported_skia/include/core/SkStrokeRec.h>
#include <ported_skia/include/core/SkPaint.h>
#include <ported_skia/include/effects/SkDashPathEffect.h>
#include <ported_skia/include/gpu/ganesh/GrRecordingContext.h>
#include <ported_skia/src/base/SkMathPriv.h>
#include <ported_skia/src/base/SkUtils.h>
#include <ported_skia/src/base/SkVx.h>
#include <ported_skia/src/gpu/ganesh/GrCaps.h>
#include <ported_skia/src/gpu/ganesh/GrClip.h>
#include <ported_skia/src/gpu/ganesh/GrAppliedClip.h>
#include <ported_skia/src/gpu/ganesh/GrProgramInfo.h>
#include <ported_skia/src/gpu/ganesh/GrShaderCaps.h>
#include <ported_skia/src/gpu/ganesh/GrStyle.h>
#include <ported_skia/src/gpu/ganesh/GrUserStencilSettings.h>
#include <ported_skia/src/gpu/ganesh/SurfaceDrawContext.h>
#include <ported_skia/src/gpu/ganesh/geometry/GrStyledShape.h>
#include <ported_skia/src/gpu/ganesh/ops/FillPathFlags.h>
#include <ported_skia/src/gpu/ganesh/ops/PathInnerTriangulateOp.h>
#include <ported_skia/src/gpu/ganesh/ops/PathStencilCoverOp.h>
#include <ported_skia/src/gpu/ganesh/ops/PathTessellateOp.h>
#include <ported_skia/src/gpu/ganesh/ops/StrokeTessellateOp.h>
#include <ported_skia/src/gpu/ganesh/ops/TessellationPathRenderer.h>
#include <ported_skia/src/gpu/ganesh/tessellate/GrPathTessellationShader.h>
#include <ported_skia/src/gpu/ganesh/tessellate/GrStrokeTessellationShader.h>
#include <ported_skia/src/gpu/tessellate/FixedCountBufferUtils.h>
#include <ported_skia/src/gpu/tessellate/Tessellation.h>
#include <ported_skia/src/gpu/tessellate/WangsFormula.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace skia_port {

namespace {

using skgpu::ganesh::PathRenderer;

static void set_error(std::string* error, const std::string& text) {
    if (error) {
        *error = text;
    }
}

static Mat3 to_mat3(const SkMatrix& m) {
    Mat3 out{};
    out.v[0] = m.get(0);
    out.v[1] = m.get(1);
    out.v[2] = m.get(2);
    out.v[3] = m.get(3);
    out.v[4] = m.get(4);
    out.v[5] = m.get(5);
    out.v[6] = m.get(6);
    out.v[7] = m.get(7);
    out.v[8] = m.get(8);
    return out;
}

static SkIRect to_sk_irect(const RectF& r) {
    return SkIRect::MakeLTRB(static_cast<int>(std::floor(r.left)),
                             static_cast<int>(std::floor(r.top)),
                             static_cast<int>(std::ceil(r.right)),
                             static_cast<int>(std::ceil(r.bottom)));
}

static GrCaps make_caps(const PatchPrepareOptions& options) {
    GrShaderCaps shaderCaps;
    shaderCaps.fInfinitySupport = options.infinitySupport;
    shaderCaps.fVertexIDSupport = options.vertexIDSupport;
    return GrCaps(shaderCaps);
}

static GrAAType to_gr_aa(AAMode aa) {
    switch (aa) {
        case AAMode::kMSAA: return GrAAType::kMSAA;
        case AAMode::kCoverage: return GrAAType::kCoverage;
        case AAMode::kNone:
        default: return GrAAType::kNone;
    }
}

static const char* aa_mode_name(AAMode aa) {
    switch (aa) {
        case AAMode::kMSAA:
            return "msaa";
        case AAMode::kCoverage:
            return "coverage";
        case AAMode::kNone:
        default:
            return "none";
    }
}

static bool validate_tessellation_aa(AAMode aa, std::string* error) {
    switch (aa) {
        case AAMode::kNone:
        case AAMode::kMSAA:
            return true;
        case AAMode::kCoverage:
            set_error(error,
                      "coverage AA is not supported by tessellation path renderer; "
                      "supported modes are none and msaa");
            return false;
        default:
            set_error(error, std::string("unknown tessellation AA mode: ") + aa_mode_name(aa));
            return false;
    }
}

static TessPrimitiveType to_plan_primitive_type(GrPrimitiveType type) {
    switch (type) {
        case GrPrimitiveType::kTriangles: return TessPrimitiveType::kTriangles;
        case GrPrimitiveType::kTriangleStrip: return TessPrimitiveType::kTriangleStrip;
        case GrPrimitiveType::kPatches: return TessPrimitiveType::kPatches;
        default: return TessPrimitiveType::kUnknown;
    }
}

static TessLayoutKind to_plan_layout_kind(GrGeometryProcessor::LayoutKind kind) {
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

static TessPrimitiveProcessorLayout layout_from_gp(const GrGeometryProcessor& gp,
                                                   bool usesVertexID) {
    TessPrimitiveProcessorLayout layout;
    layout.kind = to_plan_layout_kind(gp.layoutKind());
    layout.vertexStrideBytes = static_cast<uint32_t>(gp.vertexStride());
    layout.instanceStrideBytes = static_cast<uint32_t>(gp.instanceStride());
    layout.vertexAttributeCount = gp.vertexAttributeCount();
    layout.instanceAttributeCount = gp.instanceAttributeCount();
    layout.hasVertexAttributes = gp.hasVertexAttributes();
    layout.hasInstanceAttributes = gp.hasInstanceAttributes();
    layout.usesVertexID = usesVertexID;
    return layout;
}

static bool stencil_test_uses_clip(GrUserStencilTest test) {
    switch (test) {
        case GrUserStencilTest::kAlwaysIfInClip:
        case GrUserStencilTest::kEqualIfInClip:
        case GrUserStencilTest::kLessIfInClip:
        case GrUserStencilTest::kLEqualIfInClip:
            return true;
        default:
            return false;
    }
}

static bool stencil_test_is_always(GrUserStencilTest test) {
    return test == GrUserStencilTest::kAlways || test == GrUserStencilTest::kAlwaysIfInClip;
}

static bool stencil_op_writes(GrUserStencilOp op) {
    return op != GrUserStencilOp::kKeep;
}

static bool stencil_op_resets(GrUserStencilOp op) {
    return op == GrUserStencilOp::kZero ||
           op == GrUserStencilOp::kZeroClipBit ||
           op == GrUserStencilOp::kZeroClipAndUserBits;
}

static bool is_always_test(GrUserStencilTest test) {
    return test == GrUserStencilTest::kAlways || test == GrUserStencilTest::kAlwaysIfInClip;
}

static bool is_equal_test(GrUserStencilTest test) {
    return test == GrUserStencilTest::kEqual || test == GrUserStencilTest::kEqualIfInClip;
}

static bool is_not_equal_test(GrUserStencilTest test) {
    return test == GrUserStencilTest::kNotEqual;
}

static TessStencilTestKind to_plan_stencil_test(GrUserStencilTest test) {
    switch (test) {
        case GrUserStencilTest::kAlwaysIfInClip: return TessStencilTestKind::kAlwaysIfInClip;
        case GrUserStencilTest::kEqualIfInClip: return TessStencilTestKind::kEqualIfInClip;
        case GrUserStencilTest::kLessIfInClip: return TessStencilTestKind::kLessIfInClip;
        case GrUserStencilTest::kLEqualIfInClip: return TessStencilTestKind::kLEqualIfInClip;
        case GrUserStencilTest::kAlways: return TessStencilTestKind::kAlways;
        case GrUserStencilTest::kNever: return TessStencilTestKind::kNever;
        case GrUserStencilTest::kGreater: return TessStencilTestKind::kGreater;
        case GrUserStencilTest::kGEqual: return TessStencilTestKind::kGEqual;
        case GrUserStencilTest::kLess: return TessStencilTestKind::kLess;
        case GrUserStencilTest::kLEqual: return TessStencilTestKind::kLEqual;
        case GrUserStencilTest::kEqual: return TessStencilTestKind::kEqual;
        case GrUserStencilTest::kNotEqual: return TessStencilTestKind::kNotEqual;
    }
    return TessStencilTestKind::kAlways;
}

static TessStencilOpKind to_plan_stencil_op(GrUserStencilOp op) {
    switch (op) {
        case GrUserStencilOp::kKeep: return TessStencilOpKind::kKeep;
        case GrUserStencilOp::kZero: return TessStencilOpKind::kZero;
        case GrUserStencilOp::kReplace: return TessStencilOpKind::kReplace;
        case GrUserStencilOp::kInvert: return TessStencilOpKind::kInvert;
        case GrUserStencilOp::kIncWrap: return TessStencilOpKind::kIncWrap;
        case GrUserStencilOp::kDecWrap: return TessStencilOpKind::kDecWrap;
        case GrUserStencilOp::kIncMaybeClamp: return TessStencilOpKind::kIncMaybeClamp;
        case GrUserStencilOp::kDecMaybeClamp: return TessStencilOpKind::kDecMaybeClamp;
        case GrUserStencilOp::kZeroClipBit: return TessStencilOpKind::kZeroClipBit;
        case GrUserStencilOp::kSetClipBit: return TessStencilOpKind::kSetClipBit;
        case GrUserStencilOp::kInvertClipBit: return TessStencilOpKind::kInvertClipBit;
        case GrUserStencilOp::kSetClipAndReplaceUserBits:
            return TessStencilOpKind::kSetClipAndReplaceUserBits;
        case GrUserStencilOp::kZeroClipAndUserBits:
            return TessStencilOpKind::kZeroClipAndUserBits;
    }
    return TessStencilOpKind::kKeep;
}

static TessStencilFaceInfo stencil_face_info_from_user(
        const GrUserStencilSettings::Face& face) {
    TessStencilFaceInfo info;
    info.ref = face.fRef;
    info.test = to_plan_stencil_test(face.fTest);
    info.testMask = face.fTestMask;
    info.passOp = to_plan_stencil_op(face.fPassOp);
    info.failOp = to_plan_stencil_op(face.fFailOp);
    info.writeMask = face.fWriteMask;
    return info;
}

static TessStencilSettingsKind classify_stencil_kind(const GrUserStencilSettings& settings,
                                                     TessPlanPassKind passKind) {
    const bool twoSided = settings.isTwoSided(true);
    const auto& front = settings.fCWFace;
    const auto& back = settings.fCCWFace;

    if (twoSided &&
        is_always_test(front.fTest) &&
        is_always_test(back.fTest) &&
        front.fPassOp == GrUserStencilOp::kIncWrap &&
        back.fPassOp == GrUserStencilOp::kDecWrap) {
        return TessStencilSettingsKind::kFillOrIncrDecr;
    }

    if (!twoSided &&
        is_always_test(front.fTest) &&
        front.fPassOp == GrUserStencilOp::kInvert) {
        return TessStencilSettingsKind::kFillOrInvert;
    }

    if (!twoSided &&
        front.fPassOp == GrUserStencilOp::kKeep &&
        front.fFailOp == GrUserStencilOp::kKeep &&
        is_equal_test(front.fTest) &&
        stencil_test_uses_clip(front.fTest)) {
        return TessStencilSettingsKind::kFillIfZeroAndInClip;
    }

    if (twoSided &&
        is_not_equal_test(front.fTest) &&
        is_not_equal_test(back.fTest) &&
        front.fPassOp == GrUserStencilOp::kIncWrap &&
        back.fPassOp == GrUserStencilOp::kDecWrap) {
        return TessStencilSettingsKind::kIncrDecrStencilIfNonzero;
    }

    if (!twoSided &&
        is_not_equal_test(front.fTest) &&
        front.fPassOp == GrUserStencilOp::kZero &&
        front.fFailOp == GrUserStencilOp::kKeep) {
        return (passKind == TessPlanPassKind::kStencilFanTriangles)
                ? TessStencilSettingsKind::kInvertStencilIfNonzero
                : TessStencilSettingsKind::kTestAndReset;
    }

    if (!twoSided &&
        is_equal_test(front.fTest) &&
        front.fFailOp == GrUserStencilOp::kZero) {
        return TessStencilSettingsKind::kTestAndResetInverse;
    }

    return TessStencilSettingsKind::kUnknown;
}

static TessStencilSettingsInfo stencil_info_from_user(const GrUserStencilSettings* settings,
                                                      TessPlanPassKind passKind) {
    TessStencilSettingsInfo info;
    if (!settings || settings->isUnused()) {
        info.kind = TessStencilSettingsKind::kUnused;
        return info;
    }
    const bool twoSided = settings->isTwoSided(true);
    const auto& front = settings->fCWFace;
    const auto& back = settings->fCCWFace;
    info.kind = classify_stencil_kind(*settings, passKind);
    info.usesStencil = true;
    info.twoSided = twoSided;
    info.front = stencil_face_info_from_user(front);
    info.back = stencil_face_info_from_user(back);
    const bool frontTest = !stencil_test_is_always(front.fTest);
    const bool backTest = twoSided && !stencil_test_is_always(back.fTest);
    info.testsStencil = frontTest || backTest || stencil_test_uses_clip(front.fTest) ||
                        (twoSided && stencil_test_uses_clip(back.fTest));
    info.writesStencil = stencil_op_writes(front.fPassOp) ||
                         stencil_op_writes(front.fFailOp) ||
                         (twoSided &&
                          (stencil_op_writes(back.fPassOp) ||
                           stencil_op_writes(back.fFailOp)));
    info.resetsStencil = stencil_op_resets(front.fPassOp) ||
                         stencil_op_resets(front.fFailOp) ||
                         (twoSided &&
                          (stencil_op_resets(back.fPassOp) ||
                           stencil_op_resets(back.fFailOp)));
    info.usesClipBit = stencil_test_uses_clip(front.fTest) ||
                       (twoSided && stencil_test_uses_clip(back.fTest));
    return info;
}

static uint32_t patch_attrib_mask_from_program(const GrProgramInfo& program) {
    const GrGeometryProcessor& gp = program.geomProc();
    if (auto* shader = dynamic_cast<const GrPathTessellationShader*>(&gp)) {
        return static_cast<uint32_t>(shader->patchAttribs());
    }
    if (auto* shader = dynamic_cast<const GrStrokeTessellationShader*>(&gp)) {
        return static_cast<uint32_t>(shader->attribs());
    }
    return 0u;
}

static TessStrokeProgramInfo stroke_info_from_program(const GrProgramInfo& program) {
    TessStrokeProgramInfo info;
    const auto* shader =
            dynamic_cast<const GrStrokeTessellationShader*>(&program.geomProc());
    if (!shader) {
        return info;
    }

    info.enabled = true;
    info.hasDynamicStroke = shader->hasDynamicStroke();
    info.hairline = shader->stroke().isHairlineStyle();
    if (!info.hasDynamicStroke) {
        const float maxScale = std::abs(shader->viewMatrix().getMaxScale());
        const float strokeRadius = 0.5f * (info.hairline ? 1.0f : shader->stroke().getWidth());
        info.strokeRadius = strokeRadius;
        info.joinType = skgpu::tess::GetJoinType(shader->stroke());
        info.numRadialSegmentsPerRadian = skgpu::tess::CalcNumRadialSegmentsPerRadian(
                (info.hairline ? 1.0f : maxScale) * strokeRadius);
    }
    return info;
}

static TessProgramKind program_kind_for_pass(TessPlanOpKind opKind, TessPlanPassKind passKind) {
    switch (passKind) {
        case TessPlanPassKind::kStencilCurvePatches:
            return (opKind == TessPlanOpKind::kPathInnerTriangulateOp)
                    ? TessProgramKind::kStencilCurves
                    : TessProgramKind::kStencilPath;
        case TessPlanPassKind::kStencilWedgePatches:
            return TessProgramKind::kStencilPath;
        case TessPlanPassKind::kStencilFanTriangles:
            return (opKind == TessPlanOpKind::kPathInnerTriangulateOp)
                    ? TessProgramKind::kFanStencil
                    : TessProgramKind::kStencilFan;
        case TessPlanPassKind::kFillFanTriangles:
            return TessProgramKind::kFanFill;
        case TessPlanPassKind::kCoverHulls:
            return TessProgramKind::kCoverHulls;
        case TessPlanPassKind::kCoverBoundingBoxes:
            return TessProgramKind::kCoverBoundingBox;
        case TessPlanPassKind::kStrokePatches:
            return TessProgramKind::kStencilPath;
        case TessPlanPassKind::kUnknown:
        default:
            return TessProgramKind::kUnknown;
    }
}

static TessProgramInfo program_info_from_gr(const GrProgramInfo& program,
                                            TessPlanOpKind opKind,
                                            TessPlanPassKind passKind,
                                            bool usesVertexID) {
    TessProgramInfo info;
    info.kind = program_kind_for_pass(opKind, passKind);
    info.primitiveType = to_plan_primitive_type(program.primitiveType());
    info.numSamples = static_cast<uint32_t>(std::max(program.numSamples(), 1));
    info.stencilSettings = stencil_info_from_user(program.userStencilSettings(), passKind);
    info.layout = layout_from_gp(program.geomProc(), usesVertexID);
    info.patchAttribMask = patch_attrib_mask_from_program(program);
    info.usesVertexID = usesVertexID;
    info.hasStencilClip = program.pipeline().hasStencilClip();
    info.wireframe = program.pipeline().isWireframe();
    info.stencilOnly = program.pipeline().isStencilOnly();
    info.stroke = stroke_info_from_program(program);
    return info;
}

static Mat3 shader_matrix_from_program(const GrProgramInfo& program) {
    const GrGeometryProcessor& gp = program.geomProc();
    if (auto* tess = dynamic_cast<const GrTessellationShader*>(&gp)) {
        return to_mat3(tess->viewMatrix());
    }
    return Mat3{};
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
    const size_t baseIndex = outMesh->vertices.size();
    outMesh->vertices.reserve(baseIndex + static_cast<size_t>(vertexCount));
    outMesh->indices.reserve(baseIndex + static_cast<size_t>(vertexCount));
    for (int i = 0; i < vertexCount; ++i) {
        outMesh->vertices.push_back({points[i].x(), points[i].y()});
        outMesh->indices.push_back(static_cast<uint32_t>(baseIndex + static_cast<size_t>(i)));
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
    const size_t oldSize = outInstances->data.size();
    outInstances->data.resize(oldSize + byteSize);
    std::memcpy(outInstances->data.data() + oldSize, cpuBuffer->data() + byteOffset, byteSize);
    outInstances->strideBytes = strideBytes;
    outInstances->instanceCount += static_cast<uint32_t>(instanceCount);
    return true;
}

static bool export_fixed_vertex_buffer(const sk_sp<const GrBuffer>& buffer,
                                       uint32_t strideBytes,
                                       PatchBufferData* outPatch,
                                       std::string* error) {
    if (!outPatch) {
        set_error(error, "outPatch is null");
        return false;
    }
    if (!buffer) {
        return true;
    }
    if (strideBytes == 0) {
        set_error(error, "vertex fallback stride is zero");
        return false;
    }
    const auto* cpuBuffer = dynamic_cast<const CpuGpuBuffer*>(buffer.get());
    if (!cpuBuffer) {
        set_error(error, "Expected CpuGpuBuffer-backed fallback vertex buffer.");
        return false;
    }
    if ((cpuBuffer->size() % strideBytes) != 0) {
        set_error(error, "Fallback vertex buffer size is not divisible by stride.");
        return false;
    }

    outPatch->fixedIndexBufferTemplate.clear();
    outPatch->fixedIndexCount = 0;
    outPatch->fixedVertexBufferTemplate.resize(cpuBuffer->size());
    std::memcpy(outPatch->fixedVertexBufferTemplate.data(), cpuBuffer->data(), cpuBuffer->size());
    outPatch->fixedVertexStrideBytes = strideBytes;
    outPatch->fixedVertexCount = static_cast<uint32_t>(cpuBuffer->size() / strideBytes);
    return true;
}

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

static const char* debug_name_for_pass(TessPlanOpKind opKind, TessPlanPassKind passKind) {
    switch (opKind) {
        case TessPlanOpKind::kPathTessellateOp:
            switch (passKind) {
                case TessPlanPassKind::kStencilWedgePatches: return "PathTessellateOp.tessellateWedges";
                default: break;
            }
            break;
        case TessPlanOpKind::kPathStencilCoverOp:
            switch (passKind) {
                case TessPlanPassKind::kStencilFanTriangles: return "PathStencilCoverOp.stencilFan";
                case TessPlanPassKind::kStencilCurvePatches: return "PathStencilCoverOp.stencilCurves";
                case TessPlanPassKind::kStencilWedgePatches: return "PathStencilCoverOp.stencilWedges";
                case TessPlanPassKind::kCoverBoundingBoxes: return "PathStencilCoverOp.coverBoundingBoxes";
                default: break;
            }
            break;
        case TessPlanOpKind::kPathInnerTriangulateOp:
            switch (passKind) {
                case TessPlanPassKind::kStencilCurvePatches: return "PathInnerTriangulateOp.stencilCurves";
                case TessPlanPassKind::kStencilFanTriangles: return "PathInnerTriangulateOp.stencilFan";
                case TessPlanPassKind::kFillFanTriangles: return "PathInnerTriangulateOp.fillInnerFan";
                case TessPlanPassKind::kCoverHulls: return "PathInnerTriangulateOp.coverHulls";
                default: break;
            }
            break;
        case TessPlanOpKind::kStrokeTessellateOp:
            switch (passKind) {
                case TessPlanPassKind::kStrokePatches: return "StrokeTessellateOp.strokePatches";
                default: break;
            }
            break;
        default:
            break;
    }
    return "GaneshOp.pass";
}

static bool is_patch_pass(TessPlanPassKind kind) {
    switch (kind) {
        case TessPlanPassKind::kStencilCurvePatches:
        case TessPlanPassKind::kStencilWedgePatches:
        case TessPlanPassKind::kCoverHulls:
        case TessPlanPassKind::kStrokePatches:
            return true;
        default:
            return false;
    }
}

static bool is_fan_pass(TessPlanPassKind kind) {
    return kind == TessPlanPassKind::kStencilFanTriangles ||
           kind == TessPlanPassKind::kFillFanTriangles;
}

static bool is_bbox_pass(TessPlanPassKind kind) {
    return kind == TessPlanPassKind::kCoverBoundingBoxes;
}

static TessPlanOpKind op_kind_from_draw_op(const GrOp* op);
static TessPlanPassKind infer_plan_pass_kind(TessPlanOpKind opKind,
                                             const GrProgramInfo& program);

static TessPassPlan make_tess_pass_plan(const GrProgramInfo& program,
                                        TessPlanOpKind opKind,
                                        TessPlanPassKind passKind,
                                        bool usesVertexID) {
    TessPassPlan pass;
    pass.kind = passKind;
    pass.programInfo = program_info_from_gr(program, opKind, passKind, usesVertexID);
    pass.shaderMatrix = shader_matrix_from_program(program);
    pass.debugName = debug_name_for_pass(opKind, passKind);
    return pass;
}

static bool append_patch_data_from_draw(const CpuOpFlushState::CapturedDraw& draw,
                                        TessPassPlan* pass,
                                        std::string* error) {
    if (!pass || !draw.program) {
        return false;
    }
    const GrGeometryProcessor& gp = draw.program->geomProc();
    const uint32_t stride = static_cast<uint32_t>(gp.instanceStride());
    if (stride == 0 || draw.command.instanceCount == 0) {
        return true;
    }
    const auto* cpuBuffer = dynamic_cast<const CpuGpuBuffer*>(draw.instanceBuffer.get());
    if (!cpuBuffer) {
        set_error(error, "Expected CpuGpuBuffer-backed patch instance buffer.");
        return false;
    }
    const size_t byteOffset = static_cast<size_t>(draw.command.baseInstance) * stride;
    const size_t byteSize = static_cast<size_t>(draw.command.instanceCount) * stride;
    if (byteOffset + byteSize > cpuBuffer->size()) {
        set_error(error, "Patch instance buffer slice exceeds backing buffer size.");
        return false;
    }

    PatchBufferData& patch = pass->patchBuffer;
    if (patch.patchStrideBytes == 0) {
        patch.patchStrideBytes = stride;
    } else if (patch.patchStrideBytes != stride) {
        set_error(error, "Patch stride mismatch across captured draws.");
        return false;
    }
    const size_t dataOffset = patch.data.size();
    patch.data.resize(dataOffset + byteSize);
    std::memcpy(patch.data.data() + dataOffset, cpuBuffer->data() + byteOffset, byteSize);

    PatchBufferData::Chunk chunk;
    chunk.basePatch = draw.command.baseInstance;
    chunk.patchCount = draw.command.instanceCount;
    chunk.byteOffset = static_cast<uint32_t>(dataOffset);
    chunk.byteSize = static_cast<uint32_t>(byteSize);
    patch.chunks.push_back(chunk);
    patch.patchCount += draw.command.instanceCount;

    patch.attribMask = pass->programInfo.patchAttribMask;

    if (draw.command.kind == TessDrawCommandKind::kIndexedInstanced ||
        draw.command.kind == TessDrawCommandKind::kInstanced) {
        patch.maxFixedCountVertexCount = std::max(
                patch.maxFixedCountVertexCount,
                static_cast<int>(draw.command.elementCount));
    }

    if (pass->kind == TessPlanPassKind::kStrokePatches &&
        cpuBuffer->allocatedSizeInBytes() >= stride &&
        (cpuBuffer->allocatedSizeInBytes() % stride) == 0) {
        patch.preallocPatchCount = std::max(
                patch.preallocPatchCount,
                static_cast<uint32_t>(cpuBuffer->allocatedSizeInBytes() / stride));
    } else if (patch.preallocPatchCount < patch.patchCount) {
        patch.preallocPatchCount = patch.patchCount;
    }

    const bool wedgeMode = (patch.attribMask &
                            static_cast<uint32_t>(skgpu::tess::PatchAttribs::kFanPoint)) != 0;

    if (draw.indexBuffer && draw.vertexBuffer &&
        patch.fixedVertexBufferTemplate.empty() &&
        (draw.command.kind == TessDrawCommandKind::kIndexedInstanced)) {
        const auto* fixedVertex = dynamic_cast<const CpuGpuBuffer*>(draw.vertexBuffer.get());
        const auto* fixedIndex = dynamic_cast<const CpuGpuBuffer*>(draw.indexBuffer.get());
        if (!fixedVertex || !fixedIndex) {
            set_error(error, "Expected CpuGpuBuffer-backed fixed template buffers.");
            return false;
        }
        patch.fixedVertexBufferTemplate.resize(fixedVertex->size());
        std::memcpy(patch.fixedVertexBufferTemplate.data(),
                    fixedVertex->data(),
                    fixedVertex->size());
        patch.fixedIndexBufferTemplate.resize(fixedIndex->size());
        std::memcpy(patch.fixedIndexBufferTemplate.data(),
                    fixedIndex->data(),
                    fixedIndex->size());
        if (wedgeMode) {
            patch.fixedVertexStrideBytes =
                    static_cast<uint32_t>(skgpu::tess::FixedCountWedges::VertexBufferStride());
            patch.fixedVertexCount =
                    static_cast<uint32_t>(skgpu::tess::FixedCountWedges::VertexBufferVertexCount());
            patch.fixedIndexCount =
                    static_cast<uint32_t>(skgpu::tess::FixedCountWedges::IndexBufferSize() /
                                          sizeof(uint16_t));
        } else {
            patch.fixedVertexStrideBytes =
                    static_cast<uint32_t>(skgpu::tess::FixedCountCurves::VertexBufferStride());
            patch.fixedVertexCount =
                    static_cast<uint32_t>(skgpu::tess::FixedCountCurves::VertexBufferVertexCount());
            patch.fixedIndexCount =
                    static_cast<uint32_t>(skgpu::tess::FixedCountCurves::IndexBufferSize() /
                                          sizeof(uint16_t));
        }
    }

    if (pass->kind == TessPlanPassKind::kStrokePatches &&
        !pass->programInfo.usesVertexID &&
        draw.vertexBuffer &&
        pass->patchBuffer.fixedVertexBufferTemplate.empty()) {
        if (!export_fixed_vertex_buffer(draw.vertexBuffer,
                                        static_cast<uint32_t>(gp.vertexStride()),
                                        &patch,
                                        error)) {
            return false;
        }
    }

    if (pass->kind == TessPlanPassKind::kCoverHulls &&
        !pass->programInfo.usesVertexID &&
        draw.vertexBuffer &&
        pass->patchBuffer.fixedVertexBufferTemplate.empty()) {
        if (!export_fixed_vertex_buffer(draw.vertexBuffer,
                                        static_cast<uint32_t>(gp.vertexStride()),
                                        &patch,
                                        error)) {
            return false;
        }
    }

    if (patch.maxFixedCountVertexCount > 0 &&
        (pass->kind == TessPlanPassKind::kStencilCurvePatches ||
         pass->kind == TessPlanPassKind::kStencilWedgePatches)) {
        patch.requiredResolveLevel = resolve_level_from_fixed_vertex_count(
                patch.maxFixedCountVertexCount, wedgeMode);
    }

    return true;
}

static bool append_draw_payload_to_pass(const CpuOpFlushState::CapturedDraw& draw,
                                        TessPlanPassKind passKind,
                                        TessPassPlan* pass,
                                        std::string* error) {
    if (!pass) {
        set_error(error, "pass is null");
        return false;
    }

    if (is_patch_pass(passKind)) {
        return append_patch_data_from_draw(draw, pass, error);
    }

    if (is_fan_pass(passKind)) {
        return export_point_buffer_as_mesh(draw.vertexBuffer,
                                           static_cast<int>(draw.command.baseVertex),
                                           static_cast<int>(draw.command.elementCount),
                                           &pass->triangleMesh,
                                           error);
    }

    if (!is_bbox_pass(passKind)) {
        return true;
    }

    const uint32_t stride =
            static_cast<uint32_t>(draw.program->geomProc().instanceStride());
    if (!export_instance_buffer(draw.instanceBuffer,
                                stride,
                                static_cast<int>(draw.command.baseInstance),
                                static_cast<int>(draw.command.instanceCount),
                                &pass->instanceBuffer,
                                error)) {
        return false;
    }
    if (!pass->programInfo.usesVertexID &&
        draw.vertexBuffer &&
        pass->patchBuffer.fixedVertexBufferTemplate.empty()) {
        const uint32_t vertexStride =
                static_cast<uint32_t>(draw.program->geomProc().vertexStride());
        if (!export_fixed_vertex_buffer(draw.vertexBuffer,
                                        vertexStride,
                                        &pass->patchBuffer,
                                        error)) {
            return false;
        }
    }

    return true;
}

static void propagate_curve_patch_attrib_mask(std::vector<TessPassPlan>* passes) {
    if (!passes) {
        return;
    }

    uint32_t curveAttribMask = 0;
    for (const auto& pass : *passes) {
        if (pass.kind == TessPlanPassKind::kStencilCurvePatches &&
            pass.patchBuffer.attribMask != 0) {
            curveAttribMask = pass.patchBuffer.attribMask;
            break;
        }
    }
    if (curveAttribMask == 0) {
        return;
    }

    for (auto& pass : *passes) {
        if (pass.kind == TessPlanPassKind::kCoverHulls) {
            if (pass.patchBuffer.attribMask == 0) {
                pass.patchBuffer.attribMask = curveAttribMask;
            }
            if (pass.programInfo.patchAttribMask == 0) {
                // Hull passes reuse the curve patch instance payload. The original Skia
                // HullShader reports PatchAttribs::kNone even when the runtime instance layout
                // carries an explicit curveType float (e.g. when infinity support is disabled).
                // Our Vulkan executor derives vertex input bindings from programInfo, so keep the
                // program-side attrib mask in sync with the uploaded patch buffer layout.
                pass.programInfo.patchAttribMask = curveAttribMask;
            }
        }
    }
}

static bool collect_tess_passes_from_draws(const std::vector<CpuOpFlushState::CapturedDraw>& captured,
                                           TessPlanOpKind opKind,
                                           bool usesVertexID,
                                           std::vector<TessPassPlan>* outPasses,
                                           std::string* error) {
    if (!outPasses) {
        set_error(error, "outPasses is null");
        return false;
    }

    TessPlanPassKind lastKind = TessPlanPassKind::kUnknown;
    const GrProgramInfo* lastProgram = nullptr;

    for (const auto& draw : captured) {
        if (!draw.program) {
            continue;
        }
        TessPlanPassKind passKind = infer_plan_pass_kind(opKind, *draw.program);
        if (passKind == TessPlanPassKind::kUnknown) {
            continue;
        }

        const bool newPass =
                outPasses->empty() || passKind != lastKind || draw.program != lastProgram;
        if (newPass) {
            outPasses->push_back(
                    make_tess_pass_plan(*draw.program, opKind, passKind, usesVertexID));
            lastKind = passKind;
            lastProgram = draw.program;
        }

        TessPassPlan& pass = outPasses->back();
        pass.drawCommands.push_back(draw.command);
        if (!append_draw_payload_to_pass(draw, passKind, &pass, error)) {
            return false;
        }
    }

    propagate_curve_patch_attrib_mask(outPasses);
    return true;
}

static TessPlanOpKind execute_ganesh_tess_ops(GrRecordingContext* context,
                                              const GrCaps& caps,
                                              GrRenderTargetProxy* proxy,
                                              skgpu::ganesh::SurfaceDrawContext* sdc,
                                              const GrClip& clip,
                                              AAMode aaType,
                                              CpuOpFlushState* flushState) {
    TessPlanOpKind opKind = TessPlanOpKind::kUnknown;
    if (!context || !proxy || !sdc || !flushState) {
        return opKind;
    }

    for (const auto& opOwner : sdc->ops()) {
        if (!opOwner) {
            continue;
        }
        auto* drawOp = static_cast<GrDrawOp*>(opOwner.get());
        GrAppliedClip appliedClip(proxy->dimensions(), proxy->backingStoreDimensions());
        SkRect clippedBounds = drawOp->bounds();
        if (clip.apply(context,
                       sdc,
                       drawOp,
                       to_gr_aa(aaType),
                       &appliedClip,
                       &clippedBounds) == GrClip::Effect::kClippedOut) {
            continue;
        }
        if (opKind == TessPlanOpKind::kUnknown) {
            opKind = op_kind_from_draw_op(drawOp);
        }
        drawOp->finalize(caps, &appliedClip, GrClampType::kAuto);
        flushState->setAppliedClip(std::move(appliedClip));
        flushState->setCurrentOp(drawOp);
        drawOp->prepare(flushState);
        drawOp->execute(flushState, clippedBounds);
        flushState->clearCurrentOp();
    }

    return opKind;
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

static void configure_stroke_rec(const StrokeOptions& strokeOptions, SkStrokeRec* strokeRec) {
    if (!strokeRec) {
        return;
    }
    *strokeRec = SkStrokeRec(SkStrokeRec::kHairline_InitStyle);
    if (strokeOptions.width > 0.0f) {
        strokeRec->setStrokeStyle(strokeOptions.width);
        strokeRec->setStrokeParams(to_sk_cap(strokeOptions.cap),
                                   to_sk_join(strokeOptions.join),
                                   strokeOptions.miterLimit);
    }
}

static sk_sp<SkPathEffect> make_dash_path_effect(const StrokeOptions& strokeOptions) {
    if (strokeOptions.dashArray.empty()) {
        return nullptr;
    }
    return SkDashPathEffect::Make({strokeOptions.dashArray.data(), strokeOptions.dashArray.size()},
                                  strokeOptions.dashOffset);
}

static bool make_gr_style_from_draw_options(const PathDrawOptions& options,
                                            GrStyle* outStyle,
                                            std::string* error) {
    if (!outStyle) {
        set_error(error, "outStyle is null");
        return false;
    }
    if (!options.isStroke) {
        *outStyle = GrStyle();
        return true;
    }

    SkStrokeRec strokeRec(SkStrokeRec::kHairline_InitStyle);
    configure_stroke_rec(options.strokeOptions, &strokeRec);
    if (sk_sp<SkPathEffect> dash = make_dash_path_effect(options.strokeOptions)) {
        *outStyle = GrStyle(strokeRec, std::move(dash));
    } else if (!options.strokeOptions.dashArray.empty()) {
        set_error(error, "invalid dash intervals for SkDashPathEffect");
        return false;
    } else {
        *outStyle = GrStyle(strokeRec);
    }
    return true;
}

static bool resolve_shape_for_tessellation_renderer(const SkMatrix& viewMatrix,
                                                    skgpu::ganesh::TessellationPathRenderer* renderer,
                                                    PathRenderer::CanDrawPathArgs* canDrawArgs,
                                                    GrStyledShape* ioShape,
                                                    std::string* error) {
    if (!renderer || !canDrawArgs || !ioShape) {
        set_error(error, "resolve_shape_for_tessellation_renderer received null input");
        return false;
    }
    if (renderer->canDrawPath(*canDrawArgs) == PathRenderer::CanDrawPath::kYes) {
        return true;
    }

    const SkScalar styleScale = GrStyle::MatrixToScaleFactor(viewMatrix);
    if (styleScale <= 0.0f || !std::isfinite(styleScale)) {
        set_error(error, "invalid style scale for path-effect application");
        return false;
    }

    if (ioShape->style().pathEffect()) {
        *ioShape = ioShape->applyStyle(GrStyle::Apply::kPathEffectOnly, styleScale);
        if (ioShape->isEmpty() && !ioShape->inverseFilled()) {
            return true;
        }
        if (renderer->canDrawPath(*canDrawArgs) == PathRenderer::CanDrawPath::kYes) {
            return true;
        }
    }
    if (ioShape->style().applies()) {
        *ioShape = ioShape->applyStyle(GrStyle::Apply::kPathEffectAndStrokeRec, styleScale);
        if (ioShape->isEmpty() && !ioShape->inverseFilled()) {
            return true;
        }
        if (renderer->canDrawPath(*canDrawArgs) == PathRenderer::CanDrawPath::kYes) {
            return true;
        }
    }

    set_error(error, "TessellationPathRenderer rejected drawPath args");
    return false;
}

static TessPlanOpKind op_kind_from_draw_op(const GrOp* op) {
    if (dynamic_cast<const skgpu::ganesh::PathTessellateOp*>(op)) {
        return TessPlanOpKind::kPathTessellateOp;
    }
    if (dynamic_cast<const skgpu::ganesh::PathInnerTriangulateOp*>(op)) {
        return TessPlanOpKind::kPathInnerTriangulateOp;
    }
    if (dynamic_cast<const skgpu::ganesh::PathStencilCoverOp*>(op)) {
        return TessPlanOpKind::kPathStencilCoverOp;
    }
    if (dynamic_cast<const skgpu::ganesh::StrokeTessellateOp*>(op)) {
        return TessPlanOpKind::kStrokeTessellateOp;
    }
    return TessPlanOpKind::kUnknown;
}

static TessPlanPassKind infer_plan_pass_kind(TessPlanOpKind opKind,
                                             const GrProgramInfo& program) {
    const GrGeometryProcessor& gp = program.geomProc();
    switch (gp.layoutKind()) {
        case GrGeometryProcessor::LayoutKind::kBoundingBox:
            return TessPlanPassKind::kCoverBoundingBoxes;
        case GrGeometryProcessor::LayoutKind::kHull:
            return TessPlanPassKind::kCoverHulls;
        case GrGeometryProcessor::LayoutKind::kSimpleTriangle:
            return program.pipeline().isStencilOnly()
                    ? TessPlanPassKind::kStencilFanTriangles
                    : TessPlanPassKind::kFillFanTriangles;
        case GrGeometryProcessor::LayoutKind::kStroke:
            return TessPlanPassKind::kStrokePatches;
        case GrGeometryProcessor::LayoutKind::kMiddleOut:
        case GrGeometryProcessor::LayoutKind::kUnknown:
        default:
            break;
    }

    if (const auto* pathShader = dynamic_cast<const GrPathTessellationShader*>(&gp)) {
        return (pathShader->patchAttribs() & skgpu::tess::PatchAttribs::kFanPoint)
                ? TessPlanPassKind::kStencilWedgePatches
                : TessPlanPassKind::kStencilCurvePatches;
    }
    if (dynamic_cast<const GrStrokeTessellationShader*>(&gp)) {
        return TessPlanPassKind::kStrokePatches;
    }

    switch (opKind) {
        case TessPlanOpKind::kPathTessellateOp:
            return TessPlanPassKind::kStencilWedgePatches;
        case TessPlanOpKind::kStrokeTessellateOp:
            return TessPlanPassKind::kStrokePatches;
        case TessPlanOpKind::kPathInnerTriangulateOp:
            return TessPlanPassKind::kStencilCurvePatches;
        case TessPlanOpKind::kPathStencilCoverOp:
        case TessPlanOpKind::kUnknown:
        default:
            return TessPlanPassKind::kUnknown;
    }
}

static bool capture_path_draw_plan_ganesh_ops(const SkPath& skPath,
                                              const PathDrawOptions& options,
                                              TessCapturePlan* outPlan,
                                              std::string* error) {
    if (!outPlan) {
        set_error(error, "outPlan is null");
        return false;
    }

    *outPlan = {};
    outPlan->usedOriginalSkiaCore = true;

    if (skPath.isEmpty() && !skPath.isInverseFillType()) {
        outPlan->complete = true;
        return true;
    }

    const SkMatrix viewMatrix = to_sk_matrix(options.patchOptions.viewMatrix);
    if (!validate_tessellation_aa(options.aaType, error)) {
        return false;
    }
    if (viewMatrix.hasPerspective()) {
        set_error(error, "perspective viewMatrix is not supported by tessellation path renderer");
        return false;
    }

    GrCaps caps = make_caps(options.patchOptions);
    GrRecordingContext context(&caps);

    SkIRect clipBounds = SkIRect::MakeEmpty();
    if (options.patchOptions.hasClipConservativeBounds) {
        clipBounds = to_sk_irect(options.patchOptions.clipConservativeBounds);
    }
    SkRect devBounds = viewMatrix.mapRect(skPath.getBounds());
    int width = 1;
    int height = 1;
    if (options.patchOptions.hasClipConservativeBounds) {
        width = std::max(1, clipBounds.width());
        height = std::max(1, clipBounds.height());
    } else if (!devBounds.isEmpty()) {
        width = std::max(1, static_cast<int>(std::ceil(devBounds.width())));
        height = std::max(1, static_cast<int>(std::ceil(devBounds.height())));
    }
    const int sampleCount = (options.aaType == AAMode::kMSAA) ? 4 : 1;
    GrRenderTargetProxy proxy(width, height, sampleCount);
    GrSurfaceDrawContext sdc(&proxy);

    GrStyle style;
    if (!make_gr_style_from_draw_options(options, &style, error)) {
        return false;
    }
    GrStyledShape shape(skPath, style);

    skia_port::SimpleHardClip clip;
    if (options.patchOptions.hasClipConservativeBounds) {
        clip.setConservativeBounds(clipBounds);
    }
    if (options.patchOptions.hasStencilClip) {
        clip.setStencilClip(true);
    }
    SkIRect devBoundsI = to_sk_irect(RectF{devBounds.left(), devBounds.top(),
                                           devBounds.right(), devBounds.bottom()});
    const SkIRect* effectiveClipBounds =
            options.patchOptions.hasClipConservativeBounds ? &clipBounds : &devBoundsI;

    if (!options.patchOptions.preChopCurvesIfNecessary) {
        set_error(error,
                  "preChopCurvesIfNecessary=false is not supported by the minimal tessellation runtime");
        return false;
    }

    GrPaint paint;
    skgpu::ganesh::TessellationPathRenderer renderer;
    PathRenderer::CanDrawPathArgs canArgs;
    canArgs.fShape = &shape;
    canArgs.fViewMatrix = &viewMatrix;
    canArgs.fClipConservativeBounds = effectiveClipBounds;
    canArgs.fAAType = to_gr_aa(options.aaType);
    canArgs.fHasUserStencilSettings = false;
    canArgs.fCaps = &caps;
    canArgs.fProxy = &proxy;
    canArgs.fPaint = &paint;
    canArgs.fSurfaceProps = &sdc.surfaceProps();
    if (!resolve_shape_for_tessellation_renderer(viewMatrix,
                                                 &renderer,
                                                 &canArgs,
                                                 &shape,
                                                 error)) {
        return false;
    }
    if (shape.isEmpty() && !shape.inverseFilled()) {
        outPlan->complete = true;
        return true;
    }

    PathRenderer::DrawPathArgs drawArgs{&context,
                                        std::move(paint),
                                        &GrUserStencilSettings::kUnused,
                                        &sdc,
                                        &clip,
                                        effectiveClipBounds,
                                        &viewMatrix,
                                        &shape,
                                        to_gr_aa(options.aaType),
                                        false};

    if (!renderer.drawPath(drawArgs)) {
        set_error(error, "TessellationPathRenderer failed to draw path");
        return false;
    }

    if (sdc.ops().empty()) {
        outPlan->complete = true;
        return true;
    }

    CpuResourceProvider resourceProvider;
    CpuOpFlushState flushState(&resourceProvider, caps);
    flushState.setWriteViewSize(width, height, sampleCount);
    flushState.setUsesMSAASurface(options.aaType == AAMode::kMSAA);

    TessPlanOpKind opKind = execute_ganesh_tess_ops(
            &context, caps, &proxy, &sdc, static_cast<const GrClip&>(clip), options.aaType, &flushState);

    const auto& captured = flushState.capturedDraws();
    if (captured.empty()) {
        outPlan->complete = true;
        outPlan->opKind = opKind;
        return true;
    }

    std::vector<TessPassPlan> passes;
    if (!collect_tess_passes_from_draws(captured,
                                        opKind,
                                        caps.shaderCaps()->fVertexIDSupport,
                                        &passes,
                                        error)) {
        return false;
    }

    outPlan->opKind = opKind;
    outPlan->passes = std::move(passes);
    outPlan->complete = true;
    return true;
}

static PathDrawOptions make_fill_path_draw_options(const PatchPrepareOptions& prepareOptions) {
    PathDrawOptions options;
    options.patchOptions = prepareOptions;
    options.aaType = AAMode::kNone;
    return options;
}

static PathDrawOptions make_stroke_path_draw_options(const StrokeOptions& strokeOptions,
                                                     const PatchPrepareOptions& prepareOptions) {
    PathDrawOptions options;
    options.isStroke = true;
    options.strokeOptions = strokeOptions;
    options.patchOptions = prepareOptions;
    options.aaType = AAMode::kNone;
    return options;
}

}  // namespace

bool CapturePathTessellationPlanOriginalSkia(const Path& path,
                                             const PatchPrepareOptions& prepareOptions,
                                             TessCapturePlan* outPlan,
                                             std::string* error) {
    return CapturePathTessellationPlanOriginalSkia(ToSkPath(path),
                                                   prepareOptions,
                                                   outPlan,
                                                   error);
}

bool CapturePathTessellationPlanOriginalSkia(const SkPath& skPath,
                                             const PatchPrepareOptions& prepareOptions,
                                             TessCapturePlan* outPlan,
                                             std::string* error) {
    return capture_path_draw_plan_ganesh_ops(skPath,
                                             make_fill_path_draw_options(prepareOptions),
                                             outPlan,
                                             error);
}

bool CaptureStrokeTessellationPlanOriginalSkia(const Path& path,
                                               const StrokeOptions& strokeOptions,
                                               const PatchPrepareOptions& prepareOptions,
                                               TessCapturePlan* outPlan,
                                               std::string* error) {
    return CaptureStrokeTessellationPlanOriginalSkia(ToSkPath(path),
                                                     strokeOptions,
                                                     prepareOptions,
                                                     outPlan,
                                                     error);
}

bool CaptureStrokeTessellationPlanOriginalSkia(const SkPath& path,
                                               const StrokeOptions& strokeOptions,
                                               const PatchPrepareOptions& prepareOptions,
                                               TessCapturePlan* outPlan,
                                               std::string* error) {
    return capture_path_draw_plan_ganesh_ops(path,
                                             make_stroke_path_draw_options(strokeOptions,
                                                                           prepareOptions),
                                             outPlan,
                                             error);
}

bool CapturePathDrawPlanOriginalSkia(const Path& path,
                                     const PathDrawOptions& options,
                                     TessCapturePlan* outPlan,
                                     std::string* error) {
    return CapturePathDrawPlanOriginalSkia(ToSkPath(path), options, outPlan, error);
}

bool CapturePathDrawPlanOriginalSkia(const SkPath& path,
                                     const PathDrawOptions& options,
                                     TessCapturePlan* outPlan,
                                     std::string* error) {
    if (!validate_tessellation_aa(options.aaType, error)) {
        return false;
    }
    return capture_path_draw_plan_ganesh_ops(path, options, outPlan, error);
}

}  // namespace skia_port
