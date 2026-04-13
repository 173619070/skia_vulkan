#include "SkiaTessPlanVk.h"

#include <cstring>
#include <vector>

namespace skia_port {

namespace {

static void SetError(std::string* error, const char* msg) {
    if (error) {
        *error = msg ? msg : "";
    }
}

static void FlattenTriangleVertices(const MeshData& mesh, std::vector<uint8_t>* out) {
    if (!out) {
        return;
    }
    std::vector<float> flatVertices;
    flatVertices.reserve(mesh.vertices.size() * 2);
    for (const Vec2& vertex : mesh.vertices) {
        flatVertices.push_back(vertex.x);
        flatVertices.push_back(vertex.y);
    }
    out->resize(flatVertices.size() * sizeof(float));
    if (!flatVertices.empty()) {
        std::memcpy(out->data(), flatVertices.data(), out->size());
    }
}

static void ResetUploadBytes(VkTessPassUploadBytes* out,
                             uint32_t vertexStrideBytes,
                             uint32_t instanceStrideBytes) {
    if (!out) {
        return;
    }
    out->vertexBytes.clear();
    out->indexBytes.clear();
    out->instanceBytes.clear();
    out->route = VkTessPassUploadRoute::kInvalid;
    out->vertexStrideBytes = vertexStrideBytes;
    out->instanceStrideBytes = instanceStrideBytes;
    out->indexType = VK_INDEX_TYPE_UINT16;
}

static void ResetUploadPlanView(VkTessPassUploadPlanView* out) {
    if (!out) {
        return;
    }
    *out = {};
    out->indexType = VK_INDEX_TYPE_UINT16;
}

static bool BuildPatchPassUploadPlanView(const TessPassPlan& pass,
                                         VkTessPassUploadPlanView* out,
                                         std::string* error) {
    if (!out) {
        SetError(error, "BuildPatchPassUploadPlanView: out is null");
        return false;
    }
    ResetUploadPlanView(out);
    out->route = VkTessPassUploadRoute::kPatch;
    out->fixedVertexBytes = &pass.patchBuffer.fixedVertexBufferTemplate;
    out->fixedIndexBytes = &pass.patchBuffer.fixedIndexBufferTemplate;
    out->instanceBytes = &pass.patchBuffer.data;
    out->vertexStrideBytes = pass.patchBuffer.fixedVertexStrideBytes;
    out->instanceStrideBytes = pass.patchBuffer.patchStrideBytes;
    return true;
}

static bool BuildTrianglePassUploadPlanView(const TessPassPlan& pass,
                                            VkTessPassUploadPlanView* out,
                                            std::string* error) {
    if (!out) {
        SetError(error, "BuildTrianglePassUploadPlanView: out is null");
        return false;
    }
    ResetUploadPlanView(out);
    out->route = VkTessPassUploadRoute::kTriangles;
    out->vertexStrideBytes = sizeof(float) * 2;
    return true;
}

static bool BuildBBoxPassUploadPlanView(const TessPassPlan& pass,
                                        VkTessPassUploadPlanView* out,
                                        std::string* error) {
    if (!out) {
        SetError(error, "BuildBBoxPassUploadPlanView: out is null");
        return false;
    }
    ResetUploadPlanView(out);
    out->route = VkTessPassUploadRoute::kBoundingBox;
    out->fixedVertexBytes = &pass.patchBuffer.fixedVertexBufferTemplate;
    out->instanceBytes = &pass.instanceBuffer.data;
    out->vertexStrideBytes = pass.patchBuffer.fixedVertexStrideBytes;
    out->instanceStrideBytes = pass.instanceBuffer.strideBytes;
    return true;
}

static VkTessPassUploadRoute ClassifyUploadBytesRoute(TessPlanPassKind kind) {
    switch (kind) {
        case TessPlanPassKind::kStencilCurvePatches:
        case TessPlanPassKind::kStencilWedgePatches:
        case TessPlanPassKind::kCoverHulls:
        case TessPlanPassKind::kStrokePatches:
            return VkTessPassUploadRoute::kPatch;
        case TessPlanPassKind::kStencilFanTriangles:
        case TessPlanPassKind::kFillFanTriangles:
            return VkTessPassUploadRoute::kTriangles;
        case TessPlanPassKind::kCoverBoundingBoxes:
            return VkTessPassUploadRoute::kBoundingBox;
        case TessPlanPassKind::kUnknown:
        default:
            return VkTessPassUploadRoute::kInvalid;
    }
}

static bool BuildUploadPlanViewForRoute(VkTessPassUploadRoute route,
                                        const TessPassPlan& pass,
                                        VkTessPassUploadPlanView* out,
                                        std::string* error) {
    switch (route) {
        case VkTessPassUploadRoute::kPatch:
            return BuildPatchPassUploadPlanView(pass, out, error);
        case VkTessPassUploadRoute::kTriangles:
            return BuildTrianglePassUploadPlanView(pass, out, error);
        case VkTessPassUploadRoute::kBoundingBox:
            return BuildBBoxPassUploadPlanView(pass, out, error);
        case VkTessPassUploadRoute::kInvalid:
        default:
            SetError(error, "DescribeTessPassUploadPlanView: unsupported pass kind");
            return false;
    }
}

static void CopyUploadVector(const std::vector<uint8_t>* src,
                             std::vector<uint8_t>* dst) {
    if (!dst) {
        return;
    }
    dst->clear();
    if (src) {
        *dst = *src;
    }
}

}  // namespace

bool DescribeTessPassUploadPlanView(const TessPassPlan& pass,
                                    VkTessPassUploadPlanView* out,
                                    std::string* error) {
    if (!out) {
        SetError(error, "DescribeTessPassUploadPlanView: out is null");
        return false;
    }
    ResetUploadPlanView(out);
    const VkTessPassUploadRoute route = ClassifyUploadBytesRoute(pass.kind);
    return BuildUploadPlanViewForRoute(route, pass, out, error);
}

bool BuildTessPassUploadBytes(const TessPassPlan& pass,
                              VkTessPassUploadBytes* out,
                              std::string* error) {
    if (!out) {
        SetError(error, "BuildTessPassUploadBytes: out is null");
        return false;
    }
    *out = {};
    VkTessPassUploadPlanView view;
    if (!DescribeTessPassUploadPlanView(pass, &view, error)) {
        return false;
    }

    ResetUploadBytes(out, view.vertexStrideBytes, view.instanceStrideBytes);
    out->route = view.route;
    out->indexType = view.indexType;

    switch (view.route) {
        case VkTessPassUploadRoute::kPatch:
        case VkTessPassUploadRoute::kBoundingBox:
            CopyUploadVector(view.fixedVertexBytes, &out->vertexBytes);
            CopyUploadVector(view.fixedIndexBytes, &out->indexBytes);
            CopyUploadVector(view.instanceBytes, &out->instanceBytes);
            return true;
        case VkTessPassUploadRoute::kTriangles:
            FlattenTriangleVertices(pass.triangleMesh, &out->vertexBytes);
            return true;
        case VkTessPassUploadRoute::kInvalid:
        default:
            SetError(error, "BuildTessPassUploadBytes: unsupported pass kind");
            return false;
    }
}

}  // namespace skia_port
