#pragma once

#include "SkiaPathMeshPort.h"
#include <ported_skia/include/core/SkMatrix.h>
#include <ported_skia/include/core/SkPath.h>

#include <string>

namespace skia_port {

bool CapturePathCurveTessellatorPrepareOriginalSkia(const SkPath& path,
                                                    const SkMatrix& shaderMatrix,
                                                    const SkMatrix& pathMatrix,
                                                    bool infinitySupport,
                                                    const std::vector<Triangle>& extraTriangles,
                                                    PatchBufferData* outPatches,
                                                    std::vector<TessDrawCommand>* outStencilDraws,
                                                    std::vector<TessDrawCommand>* outHullDraws,
                                                    std::string* error = nullptr);

bool CapturePathWedgeTessellatorPrepareOriginalSkia(const SkPath& path,
                                                    const SkMatrix& shaderMatrix,
                                                    const SkMatrix& pathMatrix,
                                                    bool infinitySupport,
                                                    PatchBufferData* outPatches,
                                                    std::vector<TessDrawCommand>* outStencilDraws,
                                                    std::string* error = nullptr);

bool CapturePreparedPathStencilCoverOpOriginalSkia(const SkPath& path,
                                                   const PatchPrepareOptions& prepareOptions,
                                                   TessCapturePlan* outPlan,
                                                   std::string* error = nullptr);

bool CapturePreparedPathInnerTriangulateOpOriginalSkia(const SkPath& path,
                                                       const PatchPrepareOptions& prepareOptions,
                                                       TessCapturePlan* outPlan,
                                                       std::string* error = nullptr);

bool CaptureStrokeTessellatorPrepareOriginalSkia(const SkPath& path,
                                                 const StrokeOptions& strokeOptions,
                                                 const SkMatrix& shaderMatrix,
                                                 const SkMatrix& pathMatrix,
                                                 bool infinitySupport,
                                                 bool vertexIDSupport,
                                                 PatchBufferData* outPatches,
                                                 std::vector<TessDrawCommand>* outDraws,
                                                 std::string* error = nullptr);

bool CaptureStrokeTessellatorPrepareOriginalSkia(const SkPath& path,
                                                 const StrokeOptions& strokeOptions,
                                                 const SkMatrix& shaderMatrix,
                                                 const SkMatrix& pathMatrix,
                                                 bool infinitySupport,
                                                 PatchBufferData* outPatches,
                                                 std::vector<TessDrawCommand>* outDraws,
                                                 std::string* error = nullptr);

}  // namespace skia_port
