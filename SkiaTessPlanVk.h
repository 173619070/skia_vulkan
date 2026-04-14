#pragma once

#include "SkiaPathMeshPort.h"

#include <vulkan/vulkan.h>

#include <vector>

namespace skia_port {

enum class VkTessPassUploadRoute {
    kInvalid,
    kPatch,
    kTriangles,
    kBoundingBox,
};

struct VkTessPassUploadPlanView {
    VkTessPassUploadRoute route = VkTessPassUploadRoute::kInvalid;
    const std::vector<uint8_t>* fixedVertexBytes = nullptr;
    const std::vector<uint8_t>* fixedIndexBytes = nullptr;
    const std::vector<uint8_t>* instanceBytes = nullptr;
    uint32_t vertexStrideBytes = 0;
    uint32_t instanceStrideBytes = 0;
    VkIndexType indexType = VK_INDEX_TYPE_UINT16;
};

struct VkTessPassUploadBytes {
    VkTessPassUploadRoute route = VkTessPassUploadRoute::kInvalid;
    std::vector<uint8_t> vertexBytes;
    std::vector<uint8_t> indexBytes;
    std::vector<uint8_t> instanceBytes;
    uint32_t vertexStrideBytes = 0;
    uint32_t instanceStrideBytes = 0;
    VkIndexType indexType = VK_INDEX_TYPE_UINT16;
};

struct VkTessPassUploadOffsets {
    uint32_t vertexByteOffset = 0;
    uint32_t indexByteOffset = 0;
    uint32_t instanceByteOffset = 0;
};

// Single source of truth for "how a TessPassPlan maps to upload sources".
// SkiaTessPlanVk owns the interpretation; callers should not re-derive it from pass.kind.
bool DescribeTessPassUploadPlanView(const TessPassPlan& pass,
                                    VkTessPassUploadPlanView* out,
                                    std::string* error = nullptr);

bool BuildTessPassUploadBytes(const TessPassPlan& pass,
                              VkTessPassUploadBytes* out,
                              std::string* error = nullptr);

}  // namespace skia_port
