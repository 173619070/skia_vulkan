#include "SkiaVkTessPlanExecutor.h"
#include "SkiaVkUploadContext.h"
#include "SkiaPathMeshPort.h"
#include <ported_skia/src/gpu/tessellate/Tessellation.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace skia_port {

enum class TessExecutorShaderRoute {
    kUnknown,
    kBoundingBox,
    kHull,
    kStroke,
    kSimpleTriangles,
    kMiddleOutTriangles,
    kPatches,
};

static const char* to_string(TessLayoutKind kind) {
    switch (kind) {
        case TessLayoutKind::kMiddleOut: return "MiddleOut";
        case TessLayoutKind::kBoundingBox: return "BoundingBox";
        case TessLayoutKind::kHull: return "Hull";
        case TessLayoutKind::kStroke: return "Stroke";
        case TessLayoutKind::kUnknown:
        default:
            return "Unknown";
    }
}

static const char* to_string(TessPrimitiveType type) {
    switch (type) {
        case TessPrimitiveType::kTriangles: return "Triangles";
        case TessPrimitiveType::kTriangleStrip: return "TriangleStrip";
        case TessPrimitiveType::kPatches: return "Patches";
        case TessPrimitiveType::kUnknown:
        default:
            return "Unknown";
    }
}

static const char* to_string(TessExecutorShaderRoute route) {
    switch (route) {
        case TessExecutorShaderRoute::kBoundingBox: return "BoundingBox";
        case TessExecutorShaderRoute::kHull: return "Hull";
        case TessExecutorShaderRoute::kStroke: return "Stroke";
        case TessExecutorShaderRoute::kSimpleTriangles: return "SimpleTriangles";
        case TessExecutorShaderRoute::kMiddleOutTriangles: return "MiddleOutTriangles";
        case TessExecutorShaderRoute::kPatches: return "Patches";
        case TessExecutorShaderRoute::kUnknown:
        default:
            return "Unknown";
    }
}

static std::string describe_program(const TessProgramInfo& info,
                                    TessExecutorShaderRoute route) {
    std::string desc = "route=";
    desc += to_string(route);
    desc += " primitive=";
    desc += to_string(info.primitiveType);
    desc += " layout=";
    desc += to_string(info.layout.kind);
    desc += " usesVertexID=";
    desc += info.usesVertexID ? "1" : "0";
    desc += " stencilOnly=";
    desc += info.stencilOnly ? "1" : "0";
    desc += " usesStencil=";
    desc += info.stencilSettings.usesStencil ? "1" : "0";
    desc += " vertexStride=";
    desc += std::to_string(info.layout.vertexStrideBytes);
    desc += " instanceStride=";
    desc += std::to_string(info.layout.instanceStrideBytes);
    desc += " patchAttribMask=";
    desc += std::to_string(info.patchAttribMask);
    return desc;
}

static void FillShaderMatrix(const Mat3& m, float row0[4], float row1[4]) {
    row0[0] = m.v[0];
    row0[1] = m.v[1];
    row0[2] = m.v[2];
    row0[3] = 0.0f;
    row1[0] = m.v[3];
    row1[1] = m.v[4];
    row1[2] = m.v[5];
    row1[3] = 0.0f;
}

static bool binds_instance_at_slot_one_only(const TessProgramInfo& info) {
    return info.usesVertexID &&
           (info.layout.kind == TessLayoutKind::kBoundingBox ||
            info.layout.kind == TessLayoutKind::kHull ||
            info.layout.kind == TessLayoutKind::kStroke) &&
           info.layout.instanceStrideBytes > 0;
}

static TessExecutorShaderRoute executor_shader_route(const TessProgramInfo& info) {
    if (info.layout.kind == TessLayoutKind::kBoundingBox) {
        return TessExecutorShaderRoute::kBoundingBox;
    }
    if (info.layout.kind == TessLayoutKind::kHull) {
        return TessExecutorShaderRoute::kHull;
    }
    if (info.layout.kind == TessLayoutKind::kStroke) {
        return TessExecutorShaderRoute::kStroke;
    }
    if (info.layout.kind == TessLayoutKind::kMiddleOut &&
        info.primitiveType == TessPrimitiveType::kTriangles) {
        return TessExecutorShaderRoute::kMiddleOutTriangles;
    }
    if (info.primitiveType == TessPrimitiveType::kTriangles) {
        return TessExecutorShaderRoute::kSimpleTriangles;
    }
    if (info.primitiveType == TessPrimitiveType::kPatches) {
        return TessExecutorShaderRoute::kPatches;
    }
    return TessExecutorShaderRoute::kUnknown;
}

static const PatchBufferData::Chunk* find_patch_chunk_for_draw(
        const PatchBufferData& patch,
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

static uint32_t compute_rebased_patch_first_instance(uint32_t patchByteOffset,
                                                     uint32_t strideBytes,
                                                     const PatchBufferData::Chunk& chunk,
                                                     const TessDrawCommand& drawCmd) {
    if (strideBytes == 0) {
        return 0;
    }

    const uint64_t uploadBase = static_cast<uint64_t>(patchByteOffset) / strideBytes;
    const uint64_t chunkUploadBase = static_cast<uint64_t>(chunk.byteOffset) / strideBytes;
    const uint64_t rebasedBaseInstance =
            static_cast<uint64_t>(drawCmd.baseInstance) - static_cast<uint64_t>(chunk.basePatch);
    return static_cast<uint32_t>(uploadBase + chunkUploadBase + rebasedBaseInstance);
}

SkiaVkTessPlanExecutor::SkiaVkTessPlanExecutor() {}
SkiaVkTessPlanExecutor::~SkiaVkTessPlanExecutor() { cleanup(); }

static VkCompareOp ToVkCompareOp(TessStencilTestKind test) {
    switch (test) {
        case TessStencilTestKind::kAlwaysIfInClip:
        case TessStencilTestKind::kAlways:
            return VK_COMPARE_OP_ALWAYS;
        case TessStencilTestKind::kEqualIfInClip:
        case TessStencilTestKind::kEqual:
            return VK_COMPARE_OP_EQUAL;
        case TessStencilTestKind::kLessIfInClip:
        case TessStencilTestKind::kLess:
            return VK_COMPARE_OP_LESS;
        case TessStencilTestKind::kLEqualIfInClip:
        case TessStencilTestKind::kLEqual:
            return VK_COMPARE_OP_LESS_OR_EQUAL;
        case TessStencilTestKind::kNever:
            return VK_COMPARE_OP_NEVER;
        case TessStencilTestKind::kGreater:
            return VK_COMPARE_OP_GREATER;
        case TessStencilTestKind::kGEqual:
            return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case TessStencilTestKind::kNotEqual:
            return VK_COMPARE_OP_NOT_EQUAL;
        default:
            return VK_COMPARE_OP_ALWAYS;
    }
}

static bool IsClippedStencilTest(TessStencilTestKind test) {
    return test == TessStencilTestKind::kAlwaysIfInClip ||
           test == TessStencilTestKind::kEqualIfInClip ||
           test == TessStencilTestKind::kLessIfInClip ||
           test == TessStencilTestKind::kLEqualIfInClip;
}

static uint16_t ExecutorClipBit(int numStencilBits) {
    return static_cast<uint16_t>(1u << (numStencilBits - 1));
}

static uint16_t ExecutorUserMask(int numStencilBits) {
    return static_cast<uint16_t>(ExecutorClipBit(numStencilBits) - 1u);
}

static uint16_t ConcreteStencilWriteMask(const TessStencilFaceInfo& face,
                                         int numStencilBits) {
    const uint16_t userMask = ExecutorUserMask(numStencilBits);
    const uint16_t clipBit = ExecutorClipBit(numStencilBits);
    const int maxOp = std::max(static_cast<int>(face.passOp), static_cast<int>(face.failOp));

    if (maxOp <= static_cast<int>(TessStencilOpKind::kDecMaybeClamp)) {
        return static_cast<uint16_t>(face.writeMask & userMask);
    }
    if (maxOp <= static_cast<int>(TessStencilOpKind::kInvertClipBit)) {
        return clipBit;
    }
    return static_cast<uint16_t>(clipBit | (face.writeMask & userMask));
}

static uint16_t ConcreteStencilTestMask(const TessStencilFaceInfo& face,
                                        bool hasStencilClip,
                                        int numStencilBits) {
    const uint16_t userMask = ExecutorUserMask(numStencilBits);
    const uint16_t clipBit = ExecutorClipBit(numStencilBits);

    if (!hasStencilClip || !IsClippedStencilTest(face.test)) {
        return static_cast<uint16_t>(face.testMask & userMask);
    }
    if (face.test != TessStencilTestKind::kAlwaysIfInClip) {
        return static_cast<uint16_t>(clipBit | (face.testMask & userMask));
    }
    return clipBit;
}

static TessStencilTestKind ConcreteStencilTest(const TessStencilFaceInfo& face,
                                               bool hasStencilClip) {
    if (hasStencilClip && face.test == TessStencilTestKind::kAlwaysIfInClip) {
        return TessStencilTestKind::kEqual;
    }
    return face.test;
}

static uint16_t ConcreteStencilReference(const TessStencilFaceInfo& face,
                                         uint16_t concreteTestMask,
                                         uint16_t concreteWriteMask,
                                         int numStencilBits) {
    const uint16_t clipBit = ExecutorClipBit(numStencilBits);
    return static_cast<uint16_t>((clipBit | face.ref) & (concreteTestMask | concreteWriteMask));
}

static VkStencilOp ToVkStencilOp(TessStencilOpKind op) {
    switch (op) {
        case TessStencilOpKind::kKeep: return VK_STENCIL_OP_KEEP;
        case TessStencilOpKind::kZero:
        case TessStencilOpKind::kZeroClipBit:
        case TessStencilOpKind::kZeroClipAndUserBits:
            return VK_STENCIL_OP_ZERO;
        case TessStencilOpKind::kReplace:
        case TessStencilOpKind::kSetClipBit:
        case TessStencilOpKind::kSetClipAndReplaceUserBits:
            return VK_STENCIL_OP_REPLACE;
        case TessStencilOpKind::kInvert:
        case TessStencilOpKind::kInvertClipBit:
            return VK_STENCIL_OP_INVERT;
        case TessStencilOpKind::kIncWrap: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case TessStencilOpKind::kDecWrap: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
        case TessStencilOpKind::kIncMaybeClamp: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case TessStencilOpKind::kDecMaybeClamp: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        default:
            return VK_STENCIL_OP_KEEP;
    }
}

static VkStencilOpState MakeStencilOpStateFromFace(const TessStencilFaceInfo& face,
                                                   bool hasStencilClip,
                                                   int numStencilBits) {
    const uint16_t concreteWriteMask = ConcreteStencilWriteMask(face, numStencilBits);
    const uint16_t concreteTestMask = ConcreteStencilTestMask(face, hasStencilClip,
                                                              numStencilBits);
    const uint16_t concreteRef = ConcreteStencilReference(face,
                                                          concreteTestMask,
                                                          concreteWriteMask,
                                                          numStencilBits);
    const TessStencilTestKind concreteTest = ConcreteStencilTest(face, hasStencilClip);

    VkStencilOpState out{};
    out.failOp = ToVkStencilOp(face.failOp);
    out.passOp = ToVkStencilOp(face.passOp);
    out.depthFailOp = VK_STENCIL_OP_KEEP;
    out.compareOp = ToVkCompareOp(concreteTest);
    out.compareMask = concreteTestMask;
    out.writeMask = concreteWriteMask;
    out.reference = concreteRef;
    return out;
}

static VkStencilOpState MakeStencilOpStateBack(const TessStencilSettingsInfo& stencil,
                                               bool hasStencilClip,
                                               int numStencilBits) {
    return MakeStencilOpStateFromFace(stencil.twoSided ? stencil.back : stencil.front,
                                      hasStencilClip,
                                      numStencilBits);
}

static VkStencilOpState MakeStencilOpStateFront(const TessStencilSettingsInfo& stencil,
                                                bool hasStencilClip,
                                                int numStencilBits) {
    return MakeStencilOpStateFromFace(stencil.front, hasStencilClip, numStencilBits);
}

static bool StencilFaceEq(const TessStencilFaceInfo& a, const TessStencilFaceInfo& b) {
    return a.ref == b.ref &&
           a.test == b.test &&
           a.testMask == b.testMask &&
           a.passOp == b.passOp &&
           a.failOp == b.failOp &&
           a.writeMask == b.writeMask;
}

static bool InfoEq(const TessProgramInfo& a, const TessProgramInfo& b) {
    return a.kind == b.kind &&
           a.primitiveType == b.primitiveType &&
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

namespace {

struct PatchAttribLayout {
    bool hasFanPoint = false;
    bool hasCurveType = false;
    uint32_t fanPointOffset = 0;
    uint32_t curveTypeOffset = 0;
};

static PatchAttribLayout CalcPatchAttribLayout(uint32_t patchAttribMask) {
    using PatchAttribs = skgpu::tess::PatchAttribs;
    PatchAttribLayout layout;
    PatchAttribs attribs = static_cast<PatchAttribs>(patchAttribMask);
    uint32_t offset = static_cast<uint32_t>(sizeof(float) * 8);  // p0..p3

    if (attribs & PatchAttribs::kJoinControlPoint) {
        offset += sizeof(float) * 2;
    }
    if (attribs & PatchAttribs::kFanPoint) {
        layout.hasFanPoint = true;
        layout.fanPointOffset = offset;
        offset += sizeof(float) * 2;
    }
    if (attribs & PatchAttribs::kStrokeParams) {
        offset += sizeof(float) * 2;
    }
    if (attribs & PatchAttribs::kColor) {
        if (attribs & PatchAttribs::kWideColorIfEnabled) {
            offset += sizeof(float) * 4;
        } else {
            offset += sizeof(uint8_t) * 4;
        }
    }
    if (attribs & PatchAttribs::kPaintDepth) {
        offset += sizeof(float);
    }
    if (attribs & PatchAttribs::kExplicitCurveType) {
        layout.hasCurveType = true;
        layout.curveTypeOffset = offset;
        offset += sizeof(float);
    }
    if (attribs & PatchAttribs::kSsboIndex) {
        offset += sizeof(uint32_t);
    }

    return layout;
}

static VkPipelineShaderStageCreateInfo MakeShaderStage(VkShaderStageFlagBits stage,
                                                       VkShaderModule shader) {
    return {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            stage,
            shader,
            "main",
            nullptr};
}

static void AddBinding(std::vector<VkVertexInputBindingDescription>* bindings,
                       uint32_t binding,
                       uint32_t stride,
                       VkVertexInputRate inputRate) {
    VkVertexInputBindingDescription desc{};
    desc.binding = binding;
    desc.stride = stride;
    desc.inputRate = inputRate;
    bindings->push_back(desc);
}

static void AddAttribute(std::vector<VkVertexInputAttributeDescription>* attrs,
                         uint32_t location,
                         uint32_t binding,
                         VkFormat format,
                         uint32_t offset) {
    VkVertexInputAttributeDescription attr{};
    attr.location = location;
    attr.binding = binding;
    attr.format = format;
    attr.offset = offset;
    attrs->push_back(attr);
}

static void AddPatchPointInstanceAttributes(
        std::vector<VkVertexInputAttributeDescription>* attrs) {
    AddAttribute(attrs, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0);
    AddAttribute(attrs, 2, 1, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(float) * 4);
}

static void AddOptionalCurveTypeAttribute(
        std::vector<VkVertexInputAttributeDescription>* attrs,
        uint32_t location,
        const PatchAttribLayout& patchLayout) {
    if (patchLayout.hasCurveType) {
        AddAttribute(attrs,
                     location,
                     1,
                     VK_FORMAT_R32_SFLOAT,
                     patchLayout.curveTypeOffset);
    }
}

static uint32_t BuildPipelineStages(VkPipelineShaderStageCreateInfo stages[4],
                                    TessExecutorShaderRoute route,
                                    const PatchAttribLayout& patchLayout,
                                    VkShaderModule patchVert,
                                    VkShaderModule patchFanVert,
                                    VkShaderModule patchTesc,
                                    VkShaderModule patchTese,
                                    VkShaderModule patchFrag,
                                    VkShaderModule strokeVert,
                                    VkShaderModule strokeFrag,
                                    VkShaderModule bboxVert,
                                    VkShaderModule hullVert,
                                    VkShaderModule tessFillVert,
                                    VkShaderModule tessFillFrag) {
    switch (route) {
        case TessExecutorShaderRoute::kBoundingBox:
            stages[0] = MakeShaderStage(VK_SHADER_STAGE_VERTEX_BIT, bboxVert);
            stages[1] = MakeShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, tessFillFrag);
            return 2;
        case TessExecutorShaderRoute::kHull:
            stages[0] = MakeShaderStage(VK_SHADER_STAGE_VERTEX_BIT, hullVert);
            stages[1] = MakeShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, tessFillFrag);
            return 2;
        case TessExecutorShaderRoute::kStroke:
            stages[0] = MakeShaderStage(VK_SHADER_STAGE_VERTEX_BIT, strokeVert);
            stages[1] = MakeShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, strokeFrag);
            return 2;
        case TessExecutorShaderRoute::kMiddleOutTriangles: {
            const VkShaderModule vs = patchLayout.hasFanPoint ? patchFanVert : patchVert;
            stages[0] = MakeShaderStage(VK_SHADER_STAGE_VERTEX_BIT, vs);
            stages[1] = MakeShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, patchFrag);
            return 2;
        }
        case TessExecutorShaderRoute::kSimpleTriangles:
            stages[0] = MakeShaderStage(VK_SHADER_STAGE_VERTEX_BIT, tessFillVert);
            stages[1] = MakeShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, tessFillFrag);
            return 2;
        case TessExecutorShaderRoute::kPatches: {
            const VkShaderModule vs = patchLayout.hasFanPoint ? patchFanVert : patchVert;
            stages[0] = MakeShaderStage(VK_SHADER_STAGE_VERTEX_BIT, vs);
            stages[1] = MakeShaderStage(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, patchTesc);
            stages[2] = MakeShaderStage(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, patchTese);
            stages[3] = MakeShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, patchFrag);
            return 4;
        }
        case TessExecutorShaderRoute::kUnknown:
            return 0;
    }
    return 0;
}

static void BuildMiddleOutTriangleInput(const TessProgramInfo& info,
                                        const PatchAttribLayout& patchLayout,
                                        std::vector<VkVertexInputBindingDescription>* bindings,
                                        std::vector<VkVertexInputAttributeDescription>* attrs) {
    AddBinding(bindings, 0, info.layout.vertexStrideBytes, VK_VERTEX_INPUT_RATE_VERTEX);
    AddAttribute(attrs, 0, 0, VK_FORMAT_R32G32_SFLOAT, 0);

    if (info.layout.instanceStrideBytes == 0) {
        return;
    }

    AddBinding(bindings, 1, info.layout.instanceStrideBytes, VK_VERTEX_INPUT_RATE_INSTANCE);
    AddPatchPointInstanceAttributes(attrs);
    if (patchLayout.hasFanPoint) {
        AddAttribute(attrs, 3, 1, VK_FORMAT_R32G32_SFLOAT, patchLayout.fanPointOffset);
    }
    AddOptionalCurveTypeAttribute(attrs, 4, patchLayout);
}

static void BuildSimpleTriangleInput(const TessProgramInfo& info,
                                     std::vector<VkVertexInputBindingDescription>* bindings,
                                     std::vector<VkVertexInputAttributeDescription>* attrs) {
    AddBinding(bindings, 0, info.layout.vertexStrideBytes, VK_VERTEX_INPUT_RATE_VERTEX);
    AddAttribute(attrs, 0, 0, VK_FORMAT_R32G32_SFLOAT, 0);
}

static void BuildBoundingBoxTriangleStripInput(
        const TessProgramInfo& info,
        std::vector<VkVertexInputBindingDescription>* bindings,
        std::vector<VkVertexInputAttributeDescription>* attrs) {
    if (!info.usesVertexID && info.layout.vertexStrideBytes > 0) {
        AddBinding(bindings, 0, info.layout.vertexStrideBytes, VK_VERTEX_INPUT_RATE_VERTEX);
        AddAttribute(attrs, 0, 0, VK_FORMAT_R32G32_SFLOAT, 0);
    }

    if (info.layout.instanceStrideBytes == 0) {
        return;
    }

    AddBinding(bindings, 1, info.layout.instanceStrideBytes, VK_VERTEX_INPUT_RATE_INSTANCE);
    AddAttribute(attrs, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0);
    AddAttribute(attrs, 2, 1, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 4);
    AddAttribute(attrs, 3, 1, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(float) * 6);
}

static void BuildHullTriangleStripInput(
        const TessProgramInfo& info,
        const PatchAttribLayout& patchLayout,
        std::vector<VkVertexInputBindingDescription>* bindings,
        std::vector<VkVertexInputAttributeDescription>* attrs) {
    if (!info.usesVertexID && info.layout.vertexStrideBytes > 0) {
        AddBinding(bindings, 0, info.layout.vertexStrideBytes, VK_VERTEX_INPUT_RATE_VERTEX);
        AddAttribute(attrs, 0, 0, VK_FORMAT_R32_SFLOAT, 0);
    }

    if (info.layout.instanceStrideBytes == 0) {
        return;
    }

    AddBinding(bindings, 1, info.layout.instanceStrideBytes, VK_VERTEX_INPUT_RATE_INSTANCE);
    AddPatchPointInstanceAttributes(attrs);
    AddOptionalCurveTypeAttribute(attrs, 3, patchLayout);
}

static void BuildStrokeTriangleStripInput(
        const TessProgramInfo& info,
        const PatchAttribLayout& patchLayout,
        std::vector<VkVertexInputBindingDescription>* bindings,
        std::vector<VkVertexInputAttributeDescription>* attrs) {
    if (!info.usesVertexID && info.layout.vertexStrideBytes > 0) {
        AddBinding(bindings, 0, info.layout.vertexStrideBytes, VK_VERTEX_INPUT_RATE_VERTEX);
        AddAttribute(attrs, 0, 0, VK_FORMAT_R32_SFLOAT, 0);
    }

    if (info.layout.instanceStrideBytes == 0) {
        return;
    }

    AddBinding(bindings, 1, info.layout.instanceStrideBytes, VK_VERTEX_INPUT_RATE_INSTANCE);
    AddPatchPointInstanceAttributes(attrs);
    AddAttribute(attrs, 3, 1, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 8);
    AddOptionalCurveTypeAttribute(attrs, 4, patchLayout);
}

static void BuildPatchInput(const TessProgramInfo& info,
                            const PatchAttribLayout& patchLayout,
                            std::vector<VkVertexInputBindingDescription>* bindings,
                            std::vector<VkVertexInputAttributeDescription>* attrs) {
    AddBinding(bindings, 0, info.layout.vertexStrideBytes, VK_VERTEX_INPUT_RATE_VERTEX);
    AddBinding(bindings, 1, info.layout.instanceStrideBytes, VK_VERTEX_INPUT_RATE_INSTANCE);
    AddAttribute(attrs, 0, 0, VK_FORMAT_R32G32_SFLOAT, 0);
    AddPatchPointInstanceAttributes(attrs);
    if (patchLayout.hasFanPoint) {
        AddAttribute(attrs, 3, 1, VK_FORMAT_R32G32_SFLOAT, patchLayout.fanPointOffset);
    }
    AddOptionalCurveTypeAttribute(attrs, 4, patchLayout);
}

static void BuildVertexInput(const TessProgramInfo& info,
                             const PatchAttribLayout& patchLayout,
                             std::vector<VkVertexInputBindingDescription>* bindings,
                             std::vector<VkVertexInputAttributeDescription>* attrs) {
    if (info.layout.kind == TessLayoutKind::kMiddleOut &&
        info.primitiveType == TessPrimitiveType::kTriangles) {
        BuildMiddleOutTriangleInput(info, patchLayout, bindings, attrs);
        return;
    }

    if (info.primitiveType == TessPrimitiveType::kTriangles) {
        BuildSimpleTriangleInput(info, bindings, attrs);
        return;
    }

    if (info.primitiveType == TessPrimitiveType::kTriangleStrip) {
        switch (info.layout.kind) {
            case TessLayoutKind::kBoundingBox:
                BuildBoundingBoxTriangleStripInput(info, bindings, attrs);
                return;
            case TessLayoutKind::kHull:
                BuildHullTriangleStripInput(info, patchLayout, bindings, attrs);
                return;
            case TessLayoutKind::kStroke:
                BuildStrokeTriangleStripInput(info, patchLayout, bindings, attrs);
                return;
            default:
                return;
        }
    }

    if (info.primitiveType == TessPrimitiveType::kPatches) {
        BuildPatchInput(info, patchLayout, bindings, attrs);
    }
}

static VkPipelineVertexInputStateCreateInfo MakeVertexInputState(
        const std::vector<VkVertexInputBindingDescription>& bindings,
        const std::vector<VkVertexInputAttributeDescription>& attrs) {
    VkPipelineVertexInputStateCreateInfo vin{};
    vin.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vin.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
    vin.pVertexBindingDescriptions = bindings.empty() ? nullptr : bindings.data();
    vin.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vin.pVertexAttributeDescriptions = attrs.empty() ? nullptr : attrs.data();
    return vin;
}

static VkPrimitiveTopology ToVkPrimitiveTopology(TessPrimitiveType primitiveType) {
    switch (primitiveType) {
        case TessPrimitiveType::kTriangleStrip:
            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case TessPrimitiveType::kPatches:
            return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
        case TessPrimitiveType::kTriangles:
        case TessPrimitiveType::kUnknown:
        default:
            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

static VkPipelineInputAssemblyStateCreateInfo MakeInputAssemblyState(
        TessPrimitiveType primitiveType) {
    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = ToVkPrimitiveTopology(primitiveType);
    return ia;
}

static VkPipelineTessellationStateCreateInfo MakeTessellationState() {
    VkPipelineTessellationStateCreateInfo ts{};
    ts.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
    ts.patchControlPoints = 4;
    return ts;
}

struct ViewportStateBundle {
    VkViewport viewport{};
    VkRect2D scissor{};
    VkPipelineViewportStateCreateInfo state{};
};

static void BuildViewportState(VkExtent2D extent, ViewportStateBundle* bundle) {
    if (!bundle) {
        return;
    }
    bundle->viewport = {};
    bundle->viewport.width = static_cast<float>(extent.width);
    bundle->viewport.height = static_cast<float>(extent.height);
    bundle->viewport.minDepth = 0.0f;
    bundle->viewport.maxDepth = 1.0f;

    bundle->scissor = {};
    bundle->scissor.extent = extent;

    bundle->state = {};
    bundle->state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    bundle->state.viewportCount = 1;
    bundle->state.pViewports = &bundle->viewport;
    bundle->state.scissorCount = 1;
    bundle->state.pScissors = &bundle->scissor;
}

static VkPipelineRasterizationStateCreateInfo MakeRasterizationState(bool wireframe) {
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    return rs;
}

static VkPipelineMultisampleStateCreateInfo MakeMultisampleState(
        VkSampleCountFlagBits samples) {
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = samples;
    return ms;
}

static VkPipelineDepthStencilStateCreateInfo MakeDepthStencilState(
        const TessStencilSettingsInfo& stencil) {
    constexpr bool kHasStencilClip = false;
    constexpr int kNumStencilBits = 8;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    ds.stencilTestEnable = stencil.usesStencil ? VK_TRUE : VK_FALSE;
    ds.front = MakeStencilOpStateFront(stencil, kHasStencilClip, kNumStencilBits);
    ds.back = MakeStencilOpStateBack(stencil, kHasStencilClip, kNumStencilBits);
    return ds;
}

static VkPipelineColorBlendAttachmentState MakeColorBlendAttachmentState(bool stencilOnly) {
    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    cba.colorWriteMask = stencilOnly
            ? 0
            : (VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT);
    return cba;
}

static VkPipelineColorBlendStateCreateInfo MakeColorBlendState(
        const VkPipelineColorBlendAttachmentState* attachment) {
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = attachment;
    return cb;
}

static VkGraphicsPipelineCreateInfo MakeGraphicsPipelineCreateInfo(
        uint32_t stageCount,
        const VkPipelineShaderStageCreateInfo* stages,
        const VkPipelineVertexInputStateCreateInfo* vin,
        const VkPipelineInputAssemblyStateCreateInfo* ia,
        const VkPipelineTessellationStateCreateInfo* ts,
        const VkPipelineViewportStateCreateInfo* vpState,
        const VkPipelineRasterizationStateCreateInfo* rs,
        const VkPipelineMultisampleStateCreateInfo* ms,
        const VkPipelineDepthStencilStateCreateInfo* ds,
        const VkPipelineColorBlendStateCreateInfo* cb,
        VkPipelineLayout pipelineLayout,
        VkRenderPass renderPass) {
    VkGraphicsPipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.stageCount = stageCount;
    pi.pStages = stages;
    pi.pVertexInputState = vin;
    pi.pInputAssemblyState = ia;
    if (stageCount == 4) {
        pi.pTessellationState = ts;
    }
    pi.pViewportState = vpState;
    pi.pRasterizationState = rs;
    pi.pMultisampleState = ms;
    pi.pDepthStencilState = ds;
    pi.pColorBlendState = cb;
    pi.layout = pipelineLayout;
    pi.renderPass = renderPass;
    return pi;
}

template <typename PipelineCacheList>
static VkPipeline FindCachedPipeline(const PipelineCacheList& pipelines,
                                     const TessProgramInfo& info,
                                     bool useProbeShaderVariants) {
    for (const auto& cached : pipelines) {
        if (InfoEq(cached.info, info) &&
            cached.useProbeShaderVariants == useProbeShaderVariants) {
            return cached.pipeline;
        }
    }
    return VK_NULL_HANDLE;
}

template <typename PipelineCacheList>
static void DestroyCachedPipelines(VkDevice device, PipelineCacheList* pipelines) {
    if (device != VK_NULL_HANDLE) {
        for (auto& cache : *pipelines) {
            if (cache.pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, cache.pipeline, nullptr);
            }
        }
    }
    pipelines->clear();
}

template <typename PipelineCacheList>
static void StoreCachedPipeline(PipelineCacheList* pipelines,
                                const TessProgramInfo& info,
                                bool useProbeShaderVariants,
                                VkPipeline pipeline) {
    typename PipelineCacheList::value_type cache{};
    cache.info = info;
    cache.useProbeShaderVariants = useProbeShaderVariants;
    cache.pipeline = pipeline;
    pipelines->push_back(cache);
}

struct ExecutorProbeContext {
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    const std::vector<std::vector<TessProbePassInfo>>* passInfos = nullptr;
};

struct PipelineBuildState {
    PatchAttribLayout patchLayout{};
    TessExecutorShaderRoute route = TessExecutorShaderRoute::kUnknown;
    uint32_t stageCount = 0;
    VkPipelineShaderStageCreateInfo stages[4]{};
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attrs;
    VkPipelineVertexInputStateCreateInfo vin{};
    VkPipelineInputAssemblyStateCreateInfo ia{};
    VkPipelineTessellationStateCreateInfo ts{};
    ViewportStateBundle viewportState{};
    VkPipelineRasterizationStateCreateInfo rs{};
    VkPipelineMultisampleStateCreateInfo ms{};
    VkPipelineDepthStencilStateCreateInfo ds{};
    VkPipelineColorBlendAttachmentState cba{};
    VkPipelineColorBlendStateCreateInfo cb{};
    VkGraphicsPipelineCreateInfo pi{};
};

static void BuildPipelineFixedStates(const TessProgramInfo& info,
                                     const ExecutorContext& ctx,
                                     PipelineBuildState* outState) {
    BuildViewportState(ctx.msaaExtent, &outState->viewportState);
    outState->rs = MakeRasterizationState(info.wireframe);
    outState->ms = MakeMultisampleState(ctx.msaaSamples);
    outState->ds = MakeDepthStencilState(info.stencilSettings);
    outState->cba = MakeColorBlendAttachmentState(info.stencilOnly);
    outState->cb = MakeColorBlendState(&outState->cba);
}

static void FinalizePipelineCreateInfo(const ExecutorContext& ctx,
                                       VkPipelineLayout pipelineLayout,
                                       PipelineBuildState* outState) {
    outState->pi = MakeGraphicsPipelineCreateInfo(outState->stageCount,
                                                  outState->stages,
                                                  &outState->vin,
                                                  &outState->ia,
                                                  &outState->ts,
                                                  &outState->viewportState.state,
                                                  &outState->rs,
                                                  &outState->ms,
                                                  &outState->ds,
                                                  &outState->cb,
                                                  pipelineLayout,
                                                  ctx.renderPass);
}

static bool BuildPipelineState(const TessProgramInfo& info,
                               const ExecutorContext& ctx,
                               VkPipelineLayout pipelineLayout,
                               const ExecutorShaderModules& shaders,
                               PipelineBuildState* outState) {
    if (!outState) {
        return false;
    }

    outState->patchLayout = CalcPatchAttribLayout(info.patchAttribMask);
    outState->route = executor_shader_route(info);
    if (outState->route == TessExecutorShaderRoute::kUnknown) {
        const std::string desc = describe_program(info, outState->route);
        std::cerr << "Unsupported tessellation executor route for primitive/layout combination: "
                  << desc << std::endl;
        return false;
    }

    const VkShaderModule patchVert =
            shaders.useProbeShaderVariants && shaders.patchProbeVert != VK_NULL_HANDLE
                    ? shaders.patchProbeVert
                    : shaders.patchVert;
    const VkShaderModule patchFanVert =
            shaders.useProbeShaderVariants && shaders.patchFanProbeVert != VK_NULL_HANDLE
                    ? shaders.patchFanProbeVert
                    : shaders.patchFanVert;
    const VkShaderModule strokeVert =
            shaders.useProbeShaderVariants && shaders.strokeProbeVert != VK_NULL_HANDLE
                    ? shaders.strokeProbeVert
                    : shaders.strokeVert;
    const VkShaderModule bboxVert =
            shaders.useProbeShaderVariants && shaders.bboxProbeVert != VK_NULL_HANDLE
                    ? shaders.bboxProbeVert
                    : shaders.bboxVert;
    const VkShaderModule hullVert =
            shaders.useProbeShaderVariants && shaders.hullProbeVert != VK_NULL_HANDLE
                    ? shaders.hullProbeVert
                    : shaders.hullVert;
    const VkShaderModule tessFillVert =
            shaders.useProbeShaderVariants && shaders.tessFillProbeVert != VK_NULL_HANDLE
                    ? shaders.tessFillProbeVert
                    : shaders.tessFillVert;

    outState->stageCount = BuildPipelineStages(outState->stages,
                                               outState->route,
                                               outState->patchLayout,
                                               patchVert,
                                               patchFanVert,
                                               shaders.patchTesc,
                                               shaders.patchTese,
                                               shaders.patchFrag,
                                               strokeVert,
                                               shaders.strokeFrag,
                                               bboxVert,
                                               hullVert,
                                               tessFillVert,
                                               shaders.tessFillFrag);

    outState->bindings.clear();
    outState->attrs.clear();
    BuildVertexInput(info, outState->patchLayout, &outState->bindings, &outState->attrs);
    outState->vin = MakeVertexInputState(outState->bindings, outState->attrs);
    outState->ia = MakeInputAssemblyState(info.primitiveType);
    outState->ts = MakeTessellationState();
    BuildPipelineFixedStates(info, ctx, outState);
    FinalizePipelineCreateInfo(ctx, pipelineLayout, outState);
    return true;
}

static bool HasProbeContext(const ExecutorProbeContext& probe) {
    return probe.descriptorSet != VK_NULL_HANDLE && probe.passInfos != nullptr;
}

static bool HasProbeDescriptorSet(const ExecutorProbeContext& probe) {
    return probe.descriptorSet != VK_NULL_HANDLE;
}

static const TessProbePassInfo* TryGetProbePassInfo(const ExecutorProbeContext& probe,
                                                    size_t shapeIndex,
                                                    size_t passIndex) {
    if (!HasProbeContext(probe) ||
        shapeIndex >= probe.passInfos->size() ||
        passIndex >= (*probe.passInfos)[shapeIndex].size()) {
        return nullptr;
    }
    return &(*probe.passInfos)[shapeIndex][passIndex];
}

static uint32_t CalcStrokeFlags(const TessProgramInfo& info) {
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

static uint32_t CalcProbeFlagsTemplate(const TessProgramInfo& info) {
    uint32_t probeFlags = 0;
    if (info.layout.kind == TessLayoutKind::kMiddleOut &&
        info.primitiveType == TessPrimitiveType::kTriangles) {
        probeFlags |= 2u;
    }
    return probeFlags;
}

static void FillProbeConstants(
        TessPushConstants* pcd,
        size_t shapeIndex,
        size_t passIndex,
        uint32_t probeFlagsTemplate,
        const ExecutorProbeContext& probe) {
    pcd->probeShapeIndex = 0;
    pcd->probePassIndex = 0;
    pcd->probeVertexBase = 0;
    pcd->probeVertexCount = 0;
    pcd->probeInstanceBase = 0;
    pcd->probeInstanceCount = 0;
    pcd->probeRecordBase = 0;
    pcd->probeFlags = 0;

    const TessProbePassInfo* probeInfo = TryGetProbePassInfo(probe, shapeIndex, passIndex);
    if (!probeInfo) {
        return;
    }

    pcd->probeShapeIndex = static_cast<uint32_t>(shapeIndex);
    pcd->probePassIndex = static_cast<uint32_t>(passIndex);
    pcd->probeVertexBase = probeInfo->vertexBase;
    pcd->probeVertexCount = probeInfo->vertexCount;
    pcd->probeInstanceBase = probeInfo->instanceBase;
    pcd->probeInstanceCount = probeInfo->instanceCount;
    pcd->probeRecordBase = probeInfo->recordBase;
    pcd->probeFlags = 1u | probeFlagsTemplate;
}

static size_t pass_kind_index(TessPlanPassKind kind) {
    const size_t index = static_cast<size_t>(kind);
    if (index < kTessPlanPassKindCount) {
        return index;
    }
    return 0;
}

static bool is_valid_rect(const RectF& rect) {
    return std::isfinite(rect.left) && std::isfinite(rect.top) &&
           std::isfinite(rect.right) && std::isfinite(rect.bottom) &&
           rect.right > rect.left && rect.bottom > rect.top;
}

static RectF select_replay_bounds(const TessCapturePlan& plan) {
    if (is_valid_rect(plan.inputPath.tightBounds)) {
        return plan.inputPath.tightBounds;
    }
    return plan.inputPath.bounds;
}

static bool rects_overlap(const RectF& a, const RectF& b) {
    return !(a.right <= b.left || b.right <= a.left || a.bottom <= b.top ||
             b.bottom <= a.top);
}

static void BindExecutorDescriptorSets(VkCommandBuffer cmd,
                                       VkPipelineLayout pipelineLayout,
                                       VkDescriptorSet instanceDescriptorSet,
                                       const ExecutorProbeContext& probe) {
    const bool hasInstance = instanceDescriptorSet != VK_NULL_HANDLE;
    const bool hasProbe = HasProbeDescriptorSet(probe);
    if (!hasInstance && !hasProbe) {
        return;
    }

    if (hasInstance && hasProbe) {
        VkDescriptorSet descriptorSets[2] = {
                instanceDescriptorSet,
                probe.descriptorSet,
        };
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout,
                                0,
                                2,
                                descriptorSets,
                                0,
                                nullptr);
        return;
    }

    if (hasInstance) {
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout,
                                0,
                                1,
                                &instanceDescriptorSet,
                                0,
                                nullptr);
        return;
    }

    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout,
                            1,
                            1,
                            &probe.descriptorSet,
                            0,
                            nullptr);
}

}  // namespace

bool SkiaVkTessPlanExecutor::init(const ExecutorContext& ctx) {
    if (m_instanceDescriptorSetLayout != VK_NULL_HANDLE ||
        m_probeDescriptorSetLayout != VK_NULL_HANDLE ||
        m_instanceDescriptorPool != VK_NULL_HANDLE ||
        m_instanceDescriptorSet != VK_NULL_HANDLE ||
        !m_pipelines.empty()) {
        cleanup();
    }
    m_ctx = ctx;

    VkDescriptorSetLayoutBinding instanceBinding{};
    instanceBinding.binding = 1;
    instanceBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    instanceBinding.descriptorCount = 1;
    instanceBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                                 VK_SHADER_STAGE_FRAGMENT_BIT |
                                 VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
                                 VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;

    VkDescriptorSetLayoutCreateInfo instanceLayoutInfo{};
    instanceLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    instanceLayoutInfo.bindingCount = 1;
    instanceLayoutInfo.pBindings = &instanceBinding;

    if (vkCreateDescriptorSetLayout(m_ctx.device,
                                    &instanceLayoutInfo,
                                    nullptr,
                                    &m_instanceDescriptorSetLayout) != VK_SUCCESS) {
        cleanup();
        return false;
    }

    VkDescriptorSetLayoutBinding probeBinding{};
    probeBinding.binding = 0;
    probeBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    probeBinding.descriptorCount = 1;
    probeBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo probeLayoutInfo{};
    probeLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    probeLayoutInfo.bindingCount = 1;
    probeLayoutInfo.pBindings = &probeBinding;

    if (vkCreateDescriptorSetLayout(m_ctx.device,
                                    &probeLayoutInfo,
                                    nullptr,
                                    &m_probeDescriptorSetLayout) != VK_SUCCESS) {
        cleanup();
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

    if (vkCreateDescriptorPool(m_ctx.device,
                               &poolInfo,
                               nullptr,
                               &m_instanceDescriptorPool) != VK_SUCCESS) {
        cleanup();
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_instanceDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_instanceDescriptorSetLayout;

    if (vkAllocateDescriptorSets(m_ctx.device,
                                 &allocInfo,
                                 &m_instanceDescriptorSet) != VK_SUCCESS) {
        cleanup();
        return false;
    }
    return true;
}

void SkiaVkTessPlanExecutor::cleanup() {
    if (m_instanceDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_ctx.device, m_instanceDescriptorSetLayout, nullptr);
        m_instanceDescriptorSetLayout = VK_NULL_HANDLE;
    }
    if (m_probeDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_ctx.device, m_probeDescriptorSetLayout, nullptr);
        m_probeDescriptorSetLayout = VK_NULL_HANDLE;
    }
    if (m_instanceDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_ctx.device, m_instanceDescriptorPool, nullptr);
        m_instanceDescriptorPool = VK_NULL_HANDLE;
    }
    m_instanceDescriptorSet = VK_NULL_HANDLE;
    DestroyCachedPipelines(m_ctx.device, &m_pipelines);
    m_batches.clear();
    m_prepareStats = {};
    m_ctx = {};
}

VkPipeline SkiaVkTessPlanExecutor::getOrCreatePipeline(const TessProgramInfo& info,
                                                       VkPipelineLayout pipelineLayout,
                                                       const ExecutorShaderModules& shaders) {
    if (VkPipeline existing =
                FindCachedPipeline(m_pipelines, info, shaders.useProbeShaderVariants)) {
        return existing;
    }

    PipelineBuildState state{};
    if (!BuildPipelineState(info, m_ctx, pipelineLayout, shaders, &state)) {
        return VK_NULL_HANDLE;
    }

    VkPipeline out = VK_NULL_HANDLE;
    const VkResult result =
            vkCreateGraphicsPipelines(m_ctx.device, VK_NULL_HANDLE, 1, &state.pi, nullptr, &out);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline in VkTessPlanExecutor" << std::endl;
        return VK_NULL_HANDLE;
    }

    StoreCachedPipeline(&m_pipelines, info, shaders.useProbeShaderVariants, out);
    return out;
}

void SkiaVkTessPlanExecutor::bindSSBO(const VulkanUploadContext& uploadCtx,
                                      SkiaVkMegaBuffers& replayBuffers) {
    if (m_instanceDescriptorSet == VK_NULL_HANDLE) return;

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = replayBuffers.getInstanceSSBO();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = m_instanceDescriptorSet;
    descriptorWrite.dstBinding = 1;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(uploadCtx.device, 1, &descriptorWrite, 0, nullptr);
}

bool SkiaVkTessPlanExecutor::preparePipelines(const VulkanUploadContext& uploadCtx,
                                              const std::vector<TessCapturePlan>& plans,
                                              const std::vector<GPUPathInstance>& instances,
                                              SkiaVkMegaBuffers& replayBuffers,
                                              const std::vector<std::vector<VkTessPassUploadOffsets>>*
                                                      cachedGeometryUploadOffsets,
                                              VkPipelineLayout pipelineLayout,
                                              const ExecutorShaderModules& shaders,
                                              const std::vector<std::vector<VkTessPassUploadBytes>>* cachedPassUploads,
                                              const ExecutorCachedReplayOrders* cachedReplayOrders,
                                              const ExecutorCachedReplayPrepareStats* cachedReplayPrepareStats,
                                              const ExecutorCachedReplayBatchPlans* cachedReplayBatchPlans,
                                              const std::vector<std::vector<ExecutorReplayPassDescriptor>>*
                                                      cachedReplayPassDescriptors,
                                              const std::vector<std::vector<std::vector<ExecutorReplayDrawDescriptor>>>*
                                                      cachedReplayDrawDescriptors,
                                              bool reuseExistingUpload) {
    bool allReady = true;
    m_batches.clear();
    m_prepareStats = {};
    if (cachedReplayPrepareStats) {
        m_prepareStats.planCount = cachedReplayPrepareStats->planCount;
        m_prepareStats.passCount = cachedReplayPrepareStats->passCount;
        m_prepareStats.drawCmdCount = cachedReplayPrepareStats->drawCmdCount;
        m_prepareStats.indirectCmdCount = cachedReplayPrepareStats->indirectCmdCount;
        m_prepareStats.indexedIndirectCmdCount = cachedReplayPrepareStats->indexedIndirectCmdCount;
        m_prepareStats.globalInstanceCount = cachedReplayPrepareStats->globalInstanceCount;
        m_prepareStats.vertexUploadBytes = cachedReplayPrepareStats->vertexUploadBytes;
        m_prepareStats.indexUploadBytes = cachedReplayPrepareStats->indexUploadBytes;
        m_prepareStats.instanceUploadBytes = cachedReplayPrepareStats->instanceUploadBytes;
        m_prepareStats.passCountByKind = cachedReplayPrepareStats->passCountByKind;
        m_prepareStats.drawCmdCountByKind = cachedReplayPrepareStats->drawCmdCountByKind;
    } else {
        m_prepareStats.planCount = static_cast<uint32_t>(plans.size());
    }
    const bool strictProbeBatching = shaders.useProbeShaderVariants;
    enum class ReplayPhaseClass : uint8_t {
        kStencilPrelude,
        kColorResolve,
    };
    struct PreparedReplayDraw {
        MdiBatch batch;
        TessPlanPassKind passKind = TessPlanPassKind::kUnknown;
    };
    struct PreparedReplayPhase {
        ReplayPhaseClass phaseClass = ReplayPhaseClass::kColorResolve;
        std::vector<PreparedReplayDraw> draws;
    };
    struct PreparedReplayShape {
        RectF bounds{};
        bool hasBounds = false;
        std::vector<PreparedReplayPhase> phases;
    };
    auto failPrepare = [&](const char* message) {
        std::cerr << message << std::endl;
        if (!reuseExistingUpload) {
            replayBuffers.resetOffsets();
        }
        m_batches.clear();
        return false;
    };
    auto canAppendToBatch = [&](const MdiBatch& prev, const MdiBatch& next) {
        if (prev.pipeline != next.pipeline ||
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
        if (prev.baseSSBOIndex + prev.cmdCount != next.baseSSBOIndex) {
            return false;
        }
        if (prev.firstCmdIndex + prev.cmdCount != next.firstCmdIndex) {
            return false;
        }

        return true;
    };

    if (pipelineLayout == VK_NULL_HANDLE) {
        return failPrepare("SkiaVkTessPlanExecutor: pipelineLayout is null.");
    }
    if (plans.size() != instances.size()) {
        return failPrepare("SkiaVkTessPlanExecutor: plans/instances size mismatch.");
    }
    if (cachedPassUploads && cachedPassUploads->size() != plans.size()) {
        return failPrepare("SkiaVkTessPlanExecutor: plans/cachedPassUploads size mismatch.");
    }
    if (!cachedGeometryUploadOffsets) {
        return failPrepare(
                "SkiaVkTessPlanExecutor: cached geometry upload offsets are required.");
    }
    if (cachedGeometryUploadOffsets->size() != plans.size()) {
        return failPrepare(
                "SkiaVkTessPlanExecutor: plans/cachedGeometryUploadOffsets size mismatch.");
    }
    if (cachedReplayPassDescriptors && cachedReplayPassDescriptors->size() != plans.size()) {
        return failPrepare(
                "SkiaVkTessPlanExecutor: plans/cachedReplayPassDescriptors size mismatch.");
    }
    if (cachedReplayDrawDescriptors && cachedReplayDrawDescriptors->size() != plans.size()) {
        return failPrepare(
                "SkiaVkTessPlanExecutor: plans/cachedReplayDrawDescriptors size mismatch.");
    }
    if (reuseExistingUpload &&
        (!cachedReplayOrders || !cachedReplayPrepareStats || !cachedReplayBatchPlans ||
         !cachedReplayPassDescriptors || !cachedReplayDrawDescriptors)) {
        return failPrepare(
                "SkiaVkTessPlanExecutor: cached replay metadata is required when reusing upload.");
    }

    auto phaseClassForPass = [](TessPlanPassKind passKind) {
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
    };

    std::vector<PreparedReplayShape> preparedShapes;
    std::vector<std::vector<std::vector<PreparedReplayDraw>>> preparedDrawLookup;
    const bool useCachedReplayOrders = cachedReplayOrders != nullptr;
    if (useCachedReplayOrders) {
        preparedDrawLookup.resize(plans.size());
    } else {
        preparedShapes.reserve(plans.size());
    }

    for (size_t i = 0; i < plans.size(); ++i) {
        const TessCapturePlan& plan = plans[i];
        const GPUPathInstance& inst = instances[i];
        PreparedReplayShape* preparedShape = nullptr;
        if (useCachedReplayOrders) {
            preparedDrawLookup[i].resize(plan.passes.size());
        } else {
            preparedShapes.push_back({});
            preparedShape = &preparedShapes.back();
            preparedShape->bounds = select_replay_bounds(plan);
            preparedShape->hasBounds = is_valid_rect(preparedShape->bounds);
        }

        for (size_t passIdx = 0; passIdx < plan.passes.size(); ++passIdx) {
            const TessPassPlan& pass = plan.passes[passIdx];
            const ExecutorReplayPassDescriptor* passDescriptor = nullptr;
            if (cachedReplayPassDescriptors) {
                if ((*cachedReplayPassDescriptors)[i].size() != plan.passes.size()) {
                    return failPrepare(
                            "SkiaVkTessPlanExecutor: cachedReplayPassDescriptors per-plan size mismatch.");
                }
                passDescriptor = &(*cachedReplayPassDescriptors)[i][passIdx];
            }
            const std::vector<ExecutorReplayDrawDescriptor>* passDrawDescriptors = nullptr;
            if (cachedReplayDrawDescriptors) {
                if ((*cachedReplayDrawDescriptors)[i].size() != plan.passes.size()) {
                    return failPrepare(
                            "SkiaVkTessPlanExecutor: cachedReplayDrawDescriptors per-plan size mismatch.");
                }
                passDrawDescriptors = &(*cachedReplayDrawDescriptors)[i][passIdx];
                if (passDrawDescriptors->size() != pass.drawCommands.size()) {
                    return failPrepare(
                            "SkiaVkTessPlanExecutor: cachedReplayDrawDescriptors per-pass size mismatch.");
                }
            }
            if ((*cachedGeometryUploadOffsets)[i].size() != plan.passes.size()) {
                return failPrepare(
                        "SkiaVkTessPlanExecutor: cachedGeometryUploadOffsets per-plan size mismatch.");
            }
            const TessProgramInfo& programInfo =
                    passDescriptor ? passDescriptor->programInfo : pass.programInfo;
            const TessPlanPassKind passKind = passDescriptor ? passDescriptor->passKind : pass.kind;
            const size_t passKindIndex = pass_kind_index(passKind);
            if (!cachedReplayPrepareStats) {
                ++m_prepareStats.passCount;
                ++m_prepareStats.passCountByKind[passKindIndex];
                m_prepareStats.drawCmdCount += static_cast<uint32_t>(pass.drawCommands.size());
                m_prepareStats.drawCmdCountByKind[passKindIndex] +=
                        static_cast<uint32_t>(pass.drawCommands.size());
            }
            PreparedReplayPhase* preparedPhase = nullptr;
            if (!useCachedReplayOrders) {
                const ReplayPhaseClass phaseClass = phaseClassForPass(passKind);
                if (preparedShape->phases.empty() ||
                    preparedShape->phases.back().phaseClass != phaseClass) {
                    preparedShape->phases.push_back({});
                    preparedShape->phases.back().phaseClass = phaseClass;
                }
                preparedPhase = &preparedShape->phases.back();
            }

            if (reuseExistingUpload) {
                if (!passDescriptor || !passDrawDescriptors) {
                    return failPrepare(
                            "SkiaVkTessPlanExecutor: pass/draw descriptors are required when reusing upload.");
                }

                VkPipeline pipeline = getOrCreatePipeline(programInfo, pipelineLayout, shaders);
                if (pipeline == VK_NULL_HANDLE) {
                    allReady = false;
                    continue;
                }

                auto& reuseDraws = preparedDrawLookup[i][passIdx];
                reuseDraws.reserve(passDrawDescriptors->size());
                for (size_t cmdIdx = 0; cmdIdx < passDrawDescriptors->size(); ++cmdIdx) {
                    const ExecutorReplayDrawDescriptor& drawDescriptor = (*passDrawDescriptors)[cmdIdx];
                    MdiBatch batch;
                    batch.pipeline = pipeline;
                    batch.passKind = passKind;
                batch.firstCmdIndex = drawDescriptor.commandStreamIndex;
                batch.cmdCount = 1;
                batch.isIndexed = drawDescriptor.isIndexed;
                batch.usesReplayPatchBuffer = (passKind == TessPlanPassKind::kCoverBoundingBoxes);
                batch.baseSSBOIndex = drawDescriptor.globalInstanceOrdinal;
                batch.shapeIndex = static_cast<uint32_t>(i);
                batch.passIndex = static_cast<uint32_t>(passIdx);
                    batch.probeFlagsTemplate = passDescriptor->probeFlagsTemplate;
                    batch.hasCustomMaxResolveLevel = passDescriptor->hasCustomMaxResolveLevel;
                    batch.maxResolveLevel = passDescriptor->maxResolveLevel;

                    PreparedReplayDraw preparedDraw;
                    preparedDraw.batch = batch;
                    preparedDraw.passKind = passKind;
                    reuseDraws.push_back(preparedDraw);
                }
                continue;
            }

            VkTessPassUploadBytes localUploadBytes;
            const VkTessPassUploadBytes* uploadBytes = nullptr;
            if (cachedPassUploads) {
                if ((*cachedPassUploads)[i].size() != plan.passes.size()) {
                    return failPrepare(
                            "SkiaVkTessPlanExecutor: cachedPassUploads per-plan size mismatch.");
                }
                uploadBytes = &(*cachedPassUploads)[i][passIdx];
            } else {
                if (!BuildTessPassUploadBytes(pass, &localUploadBytes, nullptr)) {
                    allReady = false;
                    continue;
                }
                uploadBytes = &localUploadBytes;
            }
            if (!cachedReplayPrepareStats) {
                m_prepareStats.vertexUploadBytes += uploadBytes->vertexBytes.size();
                m_prepareStats.indexUploadBytes += uploadBytes->indexBytes.size();
                m_prepareStats.instanceUploadBytes += uploadBytes->instanceBytes.size();
            }

            const bool isFanPass = uploadBytes->route == VkTessPassUploadRoute::kTriangles;
            const bool isBBoxPass = uploadBytes->route == VkTessPassUploadRoute::kBoundingBox;

            VkPipeline pipeline = getOrCreatePipeline(programInfo, pipelineLayout, shaders);
            if (pipeline == VK_NULL_HANDLE) {
                allReady = false;
                continue;
            }

            const bool hasCustomMaxResolveLevel =
                    passDescriptor ? passDescriptor->hasCustomMaxResolveLevel
                                   : pass.patchBuffer.patchCount > 0;
            float passMaxResolveLevel = passDescriptor ? passDescriptor->maxResolveLevel : 0.0f;
            if (!passDescriptor && hasCustomMaxResolveLevel) {
                const float maxAllowed = static_cast<float>(skgpu::tess::kMaxResolveLevel);
                passMaxResolveLevel = std::clamp(
                        static_cast<float>(pass.patchBuffer.requiredResolveLevel), 0.0f, maxAllowed);
            }

            const VkTessPassUploadOffsets& uploadOffsets =
                    (*cachedGeometryUploadOffsets)[i][passIdx];
            const uint32_t vertexByteOffset = uploadOffsets.vertexByteOffset;
            const uint32_t indexByteOffset = uploadOffsets.indexByteOffset;
            const uint32_t patchByteOffset = uploadOffsets.instanceByteOffset;
            const uint32_t instanceStrideBytes = uploadBytes->instanceStrideBytes;
            uint32_t replayPatchByteOffset = 0;
            const bool usesReplayPatchBuffer =
                    (uploadBytes->route == VkTessPassUploadRoute::kBoundingBox);

            if (usesReplayPatchBuffer && !uploadBytes->instanceBytes.empty()) {
                if (!replayBuffers.appendPatchInstanceData(uploadBytes->instanceBytes.data(),
                                                          static_cast<uint32_t>(
                                                                  uploadBytes->instanceBytes.size()),
                                                          instanceStrideBytes,
                                                          &replayPatchByteOffset)) {
                    return failPrepare(
                            "SkiaVkTessPlanExecutor: append bbox replay patch data failed.");
                }
            }

            uint32_t localTriangleVertexCount = 0;
            uint32_t bboxInstanceByteCursor = 0;

            for (size_t cmdIdx = 0; cmdIdx < pass.drawCommands.size(); ++cmdIdx) {
                const TessDrawCommand& drawCmd = pass.drawCommands[cmdIdx];
                const ExecutorReplayDrawDescriptor* drawDescriptor =
                        passDrawDescriptors ? &(*passDrawDescriptors)[cmdIdx] : nullptr;
                const bool isIndexed =
                        drawDescriptor ? drawDescriptor->isIndexed
                                       : (drawCmd.kind == TessDrawCommandKind::kIndexedInstanced);
                GPUPathInstance localInst = passDescriptor ? passDescriptor->instanceTemplate : inst;
                if (!passDescriptor) {
                    FillShaderMatrix(pass.shaderMatrix,
                                     localInst.shaderMatrixRow0,
                                     localInst.shaderMatrixRow1);
                    localInst.strokeTessArgs[0] =
                            programInfo.stroke.numRadialSegmentsPerRadian;
                    localInst.strokeTessArgs[1] = programInfo.stroke.joinType;
                    localInst.strokeTessArgs[2] = programInfo.stroke.strokeRadius;
                    localInst.strokeTessArgs[3] =
                            static_cast<float>(CalcStrokeFlags(programInfo));
                }

                uint32_t baseInstanceOffset = 0;
                if (!replayBuffers.appendGlobalInstance(localInst, &baseInstanceOffset)) {
                    return failPrepare("SkiaVkTessPlanExecutor: appendGlobalInstance failed.");
                }
                if (!cachedReplayPrepareStats) {
                    ++m_prepareStats.globalInstanceCount;
                }

                MdiBatch batch;
                batch.pipeline = pipeline;
                batch.passKind = passKind;
                batch.isIndexed = isIndexed;
                batch.usesReplayPatchBuffer = usesReplayPatchBuffer;
                batch.baseSSBOIndex = baseInstanceOffset;
                batch.shapeIndex = static_cast<uint32_t>(i);
                batch.passIndex = static_cast<uint32_t>(passIdx);
                batch.probeFlagsTemplate = passDescriptor
                                                   ? passDescriptor->probeFlagsTemplate
                                                   : CalcProbeFlagsTemplate(programInfo);
                batch.hasCustomMaxResolveLevel = hasCustomMaxResolveLevel;
                batch.maxResolveLevel = passMaxResolveLevel;
                batch.cmdCount = 1;
                batch.firstCmdIndex = drawDescriptor
                                              ? drawDescriptor->commandStreamIndex
                                              : (isIndexed ? replayBuffers.getIndexedIndirectCmdCount()
                                                           : replayBuffers.getIndirectCmdCount());

                uint32_t vStride = uploadBytes->vertexStrideBytes;
                if (vStride == 0) vStride = 1;
                const uint32_t uploadedVertexBase = vertexByteOffset / vStride;
                const uint32_t uploadedIndexBase = indexByteOffset / sizeof(uint16_t);

                uint32_t pStride = instanceStrideBytes == 0 ? 1 : instanceStrideBytes;
                const uint32_t effectivePatchByteOffset =
                        usesReplayPatchBuffer ? replayPatchByteOffset : patchByteOffset;
                const uint32_t patchUploadBase = effectivePatchByteOffset / pStride;

                auto computeInstanceFirst = [&]() -> uint32_t {
                    if (drawDescriptor) {
                        return drawDescriptor->usesPatchBaseForFirstInstance
                                       ? patchUploadBase + drawDescriptor->firstInstanceOffset
                                       : drawDescriptor->firstInstanceOffset;
                    }
                    if (isBBoxPass) {
                        // Bounding box instances are exported as per-draw contiguous slices, so the
                        // rebased upload cursor is already the replay equivalent of drawCmd.baseInstance.
                        return (effectivePatchByteOffset + bboxInstanceByteCursor) / pStride;
                    }

                    const PatchBufferData::Chunk* chunk =
                            find_patch_chunk_for_draw(pass.patchBuffer, drawCmd);
                    if (!chunk && cmdIdx < pass.patchBuffer.chunks.size()) {
                        // Compatibility fallback only. The primary path is metadata-based lookup.
                        chunk = &pass.patchBuffer.chunks[cmdIdx];
                    }
                    if (chunk) {
                        return compute_rebased_patch_first_instance(
                                effectivePatchByteOffset, pStride, *chunk, drawCmd);
                    }

                    return (effectivePatchByteOffset / pStride) + drawCmd.baseInstance;
                };

                if (isIndexed) {
                    VkDrawIndexedIndirectCommand ind{};
                    ind.indexCount =
                            drawDescriptor ? drawDescriptor->elementCount : drawCmd.elementCount;
                    ind.instanceCount =
                            drawDescriptor ? drawDescriptor->instanceCount : drawCmd.instanceCount;
                    
                    if (drawDescriptor) {
                        ind.firstIndex = uploadedIndexBase + drawDescriptor->firstIndexOffset;
                        ind.vertexOffset =
                                static_cast<int32_t>(uploadedVertexBase) + drawDescriptor->vertexOffset;
                        ind.firstInstance = computeInstanceFirst();
                    } else if (isFanPass) {
                        ind.firstIndex = uploadedIndexBase;
                        ind.vertexOffset = static_cast<int32_t>(uploadedVertexBase + localTriangleVertexCount);
                        ind.firstInstance = 0;
                    } else {
                        ind.firstIndex = uploadedIndexBase + drawCmd.baseIndex;
                        ind.vertexOffset = static_cast<int32_t>(uploadedVertexBase + drawCmd.baseVertex);
                        ind.firstInstance = computeInstanceFirst();
                    }
                    if (!replayBuffers.appendDrawIndexedIndirectCmd(ind)) {
                        return failPrepare(
                                "SkiaVkTessPlanExecutor: appendDrawIndexedIndirectCmd failed.");
                    }
                    if (!cachedReplayPrepareStats) {
                        ++m_prepareStats.indexedIndirectCmdCount;
                    }
                } else {
                    VkDrawIndirectCommand ind{};
                    ind.vertexCount =
                            drawDescriptor ? drawDescriptor->elementCount : drawCmd.elementCount;
                    ind.instanceCount =
                            drawDescriptor ? drawDescriptor->instanceCount : drawCmd.instanceCount;
                    
                    if (drawDescriptor) {
                        ind.firstVertex = uploadedVertexBase + drawDescriptor->firstVertexOffset;
                        ind.firstInstance = computeInstanceFirst();
                    } else if (isFanPass) {
                        ind.firstVertex = uploadedVertexBase + localTriangleVertexCount;
                        ind.firstInstance = 0;
                    } else {
                        ind.firstVertex = uploadedVertexBase + drawCmd.baseVertex;
                        ind.firstInstance = computeInstanceFirst();
                    }
                    if (!replayBuffers.appendDrawIndirectCmd(ind)) {
                        return failPrepare(
                                "SkiaVkTessPlanExecutor: appendDrawIndirectCmd failed.");
                    }
                    if (!cachedReplayPrepareStats) {
                        ++m_prepareStats.indirectCmdCount;
                    }
                }
                
                if (isFanPass) {
                    localTriangleVertexCount += drawCmd.elementCount;
                }
                if (isBBoxPass && drawCmd.instanceCount > 0) {
                    bboxInstanceByteCursor += drawCmd.instanceCount * pStride;
                }

                PreparedReplayDraw preparedDraw;
                preparedDraw.batch = batch;
                preparedDraw.passKind = passKind;
                if (useCachedReplayOrders) {
                    preparedDrawLookup[i][passIdx].push_back(preparedDraw);
                } else {
                    preparedPhase->draws.push_back(preparedDraw);
                }
            }
        }
    }

    if (!allReady) {
        if (!reuseExistingUpload) {
            replayBuffers.resetOffsets();
        }
        m_batches.clear();
        return false;
    }

    auto emitPreparedDraw = [&](const PreparedReplayDraw& preparedDraw) {
        const size_t passKindIndex = pass_kind_index(preparedDraw.passKind);
        if (!m_batches.empty() && canAppendToBatch(m_batches.back(), preparedDraw.batch)) {
            m_batches.back().cmdCount++;
            ++m_prepareStats.mergedDrawCmdCount;
        } else {
            m_batches.push_back(preparedDraw.batch);
            ++m_prepareStats.batchCount;
            ++m_prepareStats.batchCountByKind[passKindIndex];
        }
    };

    if (useCachedReplayOrders) {
        const std::vector<ExecutorReplayDrawRef>& drawRefs =
                strictProbeBatching ? cachedReplayOrders->originalDrawRefs
                                    : cachedReplayOrders->windowedDrawRefs;
        const std::vector<ExecutorReplayBatchSeed>* batchSeeds = nullptr;
        if (cachedReplayBatchPlans) {
            batchSeeds = strictProbeBatching ? &cachedReplayBatchPlans->originalBatches
                                             : &cachedReplayBatchPlans->windowedBatches;
        }

        if (batchSeeds) {
            for (const ExecutorReplayBatchSeed& batchSeed : *batchSeeds) {
                if (batchSeed.drawCount == 0) {
                    return failPrepare("SkiaVkTessPlanExecutor: replay batch seed draw count is zero.");
                }
                if (batchSeed.firstDrawRefIndex >= drawRefs.size() ||
                    static_cast<size_t>(batchSeed.firstDrawRefIndex) + batchSeed.drawCount >
                            drawRefs.size()) {
                    return failPrepare(
                            "SkiaVkTessPlanExecutor: replay batch seed range is out of bounds.");
                }

                const ExecutorReplayDrawRef& firstDrawRef = drawRefs[batchSeed.firstDrawRefIndex];
                if (firstDrawRef.shapeIndex >= preparedDrawLookup.size()) {
                    return failPrepare("SkiaVkTessPlanExecutor: replay batch seed shape index out of range.");
                }
                const auto& shapeDraws = preparedDrawLookup[firstDrawRef.shapeIndex];
                if (firstDrawRef.passIndex >= shapeDraws.size()) {
                    return failPrepare("SkiaVkTessPlanExecutor: replay batch seed pass index out of range.");
                }
                const auto& passDraws = shapeDraws[firstDrawRef.passIndex];
                if (firstDrawRef.drawIndex >= passDraws.size()) {
                    return failPrepare("SkiaVkTessPlanExecutor: replay batch seed draw index out of range.");
                }

                MdiBatch batch = passDraws[firstDrawRef.drawIndex].batch;
                batch.cmdCount = batchSeed.drawCount;
                const size_t passKindIndex = pass_kind_index(batch.passKind);
                m_batches.push_back(batch);
                ++m_prepareStats.batchCount;
                ++m_prepareStats.batchCountByKind[passKindIndex];
                m_prepareStats.mergedDrawCmdCount += (batchSeed.drawCount - 1);
            }

            bindSSBO(uploadCtx, replayBuffers);
            return true;
        }

        for (const ExecutorReplayDrawRef& drawRef : drawRefs) {
            if (drawRef.shapeIndex >= preparedDrawLookup.size()) {
                return failPrepare("SkiaVkTessPlanExecutor: replay draw ref shape index out of range.");
            }
            const auto& shapeDraws = preparedDrawLookup[drawRef.shapeIndex];
            if (drawRef.passIndex >= shapeDraws.size()) {
                return failPrepare("SkiaVkTessPlanExecutor: replay draw ref pass index out of range.");
            }
            const auto& passDraws = shapeDraws[drawRef.passIndex];
            if (drawRef.drawIndex >= passDraws.size()) {
                return failPrepare("SkiaVkTessPlanExecutor: replay draw ref draw index out of range.");
            }
            emitPreparedDraw(passDraws[drawRef.drawIndex]);
        }

        bindSSBO(uploadCtx, replayBuffers);
        return true;
    }

    auto emitShapeOriginalOrder = [&](const PreparedReplayShape& preparedShape) {
        for (const PreparedReplayPhase& phase : preparedShape.phases) {
            for (const PreparedReplayDraw& preparedDraw : phase.draws) {
                emitPreparedDraw(preparedDraw);
            }
        }
    };

    auto canJoinPhaseWindow = [&](size_t begin, size_t endExclusive, size_t candidateIndex) {
        if (candidateIndex >= preparedShapes.size()) {
            return false;
        }
        const PreparedReplayShape& candidate = preparedShapes[candidateIndex];
        if (!candidate.hasBounds) {
            return false;
        }
        for (size_t shapeIndex = begin; shapeIndex < endExclusive; ++shapeIndex) {
            const PreparedReplayShape& existing = preparedShapes[shapeIndex];
            if (!existing.hasBounds || rects_overlap(existing.bounds, candidate.bounds)) {
                return false;
            }
        }
        return true;
    };

    if (strictProbeBatching) {
        for (const PreparedReplayShape& preparedShape : preparedShapes) {
            emitShapeOriginalOrder(preparedShape);
        }
    } else {
        size_t shapeIndex = 0;
        while (shapeIndex < preparedShapes.size()) {
            size_t windowEnd = shapeIndex + 1;
            while (windowEnd < preparedShapes.size() &&
                   canJoinPhaseWindow(shapeIndex, windowEnd, windowEnd)) {
                ++windowEnd;
            }

            if (windowEnd - shapeIndex <= 1) {
                emitShapeOriginalOrder(preparedShapes[shapeIndex]);
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
                        const PreparedReplayShape& preparedShape = preparedShapes[windowShape];
                        if (phaseIndex >= preparedShape.phases.size()) {
                            continue;
                        }
                        const PreparedReplayPhase& preparedPhase =
                                preparedShape.phases[phaseIndex];
                        if (preparedPhase.phaseClass != phaseClass) {
                            continue;
                        }
                        for (const PreparedReplayDraw& preparedDraw : preparedPhase.draws) {
                            emitPreparedDraw(preparedDraw);
                        }
                    }
                }
            }

            shapeIndex = windowEnd;
        }
    }

    bindSSBO(uploadCtx, replayBuffers);

    return true;
}

void SkiaVkTessPlanExecutor::resetPreparedBatches() {
    m_batches.clear();
    m_prepareStats = {};
}

static void BindPassVertexBuffers(VkCommandBuffer cmd,
                                  SkiaVkMegaBuffers& geometryBuffers,
                                  SkiaVkMegaBuffers& replayBuffers,
                                  bool useReplayPatchBuffer) {
    VkBuffer buffers[2] = {
            geometryBuffers.getTriangleVertexBuffer(),
            useReplayPatchBuffer ? replayBuffers.getPatchBuffer() : geometryBuffers.getPatchBuffer()};
    VkDeviceSize offsets[2] = {0, 0};

    if (buffers[0] != VK_NULL_HANDLE && buffers[1] != VK_NULL_HANDLE) {
        vkCmdBindVertexBuffers(cmd, 0, 2, buffers, offsets);
    } else if (buffers[1] != VK_NULL_HANDLE) {
        vkCmdBindVertexBuffers(cmd, 1, 1, &buffers[1], offsets);
    } else if (buffers[0] != VK_NULL_HANDLE) {
        vkCmdBindVertexBuffers(cmd, 0, 1, &buffers[0], offsets);
    }
}

void SkiaVkTessPlanExecutor::executeBatches(
        VkCommandBuffer cmd,
        SkiaVkMegaBuffers& geometryBuffers,
        SkiaVkMegaBuffers& replayBuffers,
        VkPipelineLayout pipelineLayout,
        const TessPushConstants& pc,
        uint32_t firstBatchIndex,
        uint32_t batchCount,
        VkPipeline& inOutPipeline,
        bool& inOutHasBoundVertexBuffers,
        bool& inOutUsesReplayPatchBuffer,
        VkDescriptorSet probeDescriptorSet,
        const std::vector<std::vector<TessProbePassInfo>>* probePasses) {
    ExecutorProbeContext probe{};
    probe.descriptorSet = probeDescriptorSet;
    probe.passInfos = probePasses;
    BindExecutorDescriptorSets(cmd, pipelineLayout, m_instanceDescriptorSet, probe);

    const uint32_t endBatchIndex = firstBatchIndex + batchCount;
    for (uint32_t i = firstBatchIndex; i < endBatchIndex && i < m_batches.size(); ++i) {
        const MdiBatch& batch = m_batches[i];
        TessPushConstants localPc = pc;
        const float maxAllowed = static_cast<float>(skgpu::tess::kMaxResolveLevel);
        localPc.maxResolveLevel = batch.hasCustomMaxResolveLevel
                                          ? batch.maxResolveLevel
                                          : std::clamp(localPc.maxResolveLevel, 0.0f, maxAllowed);
        localPc.baseCmdIndex = batch.baseSSBOIndex;
        FillProbeConstants(&localPc,
                           batch.shapeIndex,
                           batch.passIndex,
                           batch.probeFlagsTemplate,
                           probe);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(TessPushConstants), &localPc);

        if (inOutPipeline != batch.pipeline ||
            !inOutHasBoundVertexBuffers ||
            inOutUsesReplayPatchBuffer != batch.usesReplayPatchBuffer) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, batch.pipeline);
            inOutPipeline = batch.pipeline;
            BindPassVertexBuffers(
                    cmd, geometryBuffers, replayBuffers, batch.usesReplayPatchBuffer);
            inOutHasBoundVertexBuffers = true;
            inOutUsesReplayPatchBuffer = batch.usesReplayPatchBuffer;
        }

        if (batch.isIndexed) {
            VkBuffer indexBuffer = geometryBuffers.getTriangleIndexBuffer();
            if (indexBuffer != VK_NULL_HANDLE) {
                vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
            }
            vkCmdDrawIndexedIndirect(cmd,
                                     replayBuffers.getIndexedIndirectCmdBuffer(),
                                     batch.firstCmdIndex * sizeof(VkDrawIndexedIndirectCommand),
                                     batch.cmdCount,
                                     sizeof(VkDrawIndexedIndirectCommand));
        } else {
            vkCmdDrawIndirect(cmd,
                              replayBuffers.getIndirectCmdBuffer(),
                              batch.firstCmdIndex * sizeof(VkDrawIndirectCommand),
                              batch.cmdCount,
                              sizeof(VkDrawIndirectCommand));
        }
    }
}

void SkiaVkTessPlanExecutor::execute(
        VkCommandBuffer cmd,
        SkiaVkMegaBuffers& geometryBuffers,
        SkiaVkMegaBuffers& replayBuffers,
        VkPipelineLayout pipelineLayout,
        const TessPushConstants& pc,
        VkDescriptorSet probeDescriptorSet,
        const std::vector<std::vector<TessProbePassInfo>>* probePasses) {
    VkPipeline currentPipeline = VK_NULL_HANDLE;
    bool hasBoundVertexBuffers = false;
    bool currentUsesReplayPatchBuffer = false;
    executeBatches(cmd, geometryBuffers, replayBuffers, pipelineLayout, pc,
                   0, static_cast<uint32_t>(m_batches.size()),
                   currentPipeline, hasBoundVertexBuffers, currentUsesReplayPatchBuffer,
                   probeDescriptorSet, probePasses);
}

} // namespace skia_port
