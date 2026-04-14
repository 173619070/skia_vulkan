#pragma once

#include <ported_skia/src/gpu/ganesh/tessellate/PathTessellator.h>
#include <ported_skia/src/gpu/ganesh/tessellate/StrokeTessellator.h>

namespace skia_port {

namespace detail {

struct PathTessellatorMemberAccess : skgpu::ganesh::PathTessellator {
    using skgpu::ganesh::PathTessellator::fFixedIndexBuffer;
    using skgpu::ganesh::PathTessellator::fFixedVertexBuffer;
    using skgpu::ganesh::PathTessellator::fMaxVertexCount;
    using skgpu::ganesh::PathTessellator::fVertexChunkArray;
};

struct StrokeTessellatorMemberAccess : skgpu::ganesh::StrokeTessellator {
    using skgpu::ganesh::StrokeTessellator::fAttribs;
    using skgpu::ganesh::StrokeTessellator::fVertexBufferIfNoIDSupport;
    using skgpu::ganesh::StrokeTessellator::fVertexChunkArray;
    using skgpu::ganesh::StrokeTessellator::fVertexCount;
};

}  // namespace detail

inline const GrVertexChunkArray& path_tessellator_vertex_chunks(
        const skgpu::ganesh::PathTessellator& tessellator) {
    static constexpr auto kMember = &detail::PathTessellatorMemberAccess::fVertexChunkArray;
    return tessellator.*kMember;
}

inline int path_tessellator_max_vertex_count(const skgpu::ganesh::PathTessellator& tessellator) {
    static constexpr auto kMember = &detail::PathTessellatorMemberAccess::fMaxVertexCount;
    return tessellator.*kMember;
}

inline const sk_sp<const GrGpuBuffer>& path_tessellator_fixed_vertex_buffer(
        const skgpu::ganesh::PathTessellator& tessellator) {
    static constexpr auto kMember = &detail::PathTessellatorMemberAccess::fFixedVertexBuffer;
    return tessellator.*kMember;
}

inline const sk_sp<const GrGpuBuffer>& path_tessellator_fixed_index_buffer(
        const skgpu::ganesh::PathTessellator& tessellator) {
    static constexpr auto kMember = &detail::PathTessellatorMemberAccess::fFixedIndexBuffer;
    return tessellator.*kMember;
}

inline skgpu::ganesh::StrokeTessellator::PatchAttribs stroke_tessellator_patch_attribs(
        const skgpu::ganesh::StrokeTessellator& tessellator) {
    static constexpr auto kMember = &detail::StrokeTessellatorMemberAccess::fAttribs;
    return tessellator.*kMember;
}

inline const GrVertexChunkArray& stroke_tessellator_vertex_chunks(
        const skgpu::ganesh::StrokeTessellator& tessellator) {
    static constexpr auto kMember = &detail::StrokeTessellatorMemberAccess::fVertexChunkArray;
    return tessellator.*kMember;
}

inline int stroke_tessellator_max_vertex_count(
        const skgpu::ganesh::StrokeTessellator& tessellator) {
    static constexpr auto kMember = &detail::StrokeTessellatorMemberAccess::fVertexCount;
    return tessellator.*kMember;
}

inline const sk_sp<const GrGpuBuffer>& stroke_tessellator_fixed_vertex_buffer(
        const skgpu::ganesh::StrokeTessellator& tessellator) {
    static constexpr auto kMember =
            &detail::StrokeTessellatorMemberAccess::fVertexBufferIfNoIDSupport;
    return tessellator.*kMember;
}

}  // namespace skia_port
