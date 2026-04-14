#include "SkiaPathMeshPort.h"

#include <ported_skia/include/core/SkPathBuilder.h>
#include <ported_skia/include/core/SkRRect.h>
#include <ported_skia/src/core/SkPathPriv.h>

namespace skia_port {

namespace {

bool same_scalars(const float* a, const float* b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

bool same_points(const SkPoint* a, const SkPoint* b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

bool same_verbs(const SkPathVerb* a, const SkPathVerb* b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

bool paths_match_exactly(const SkPath& a, const SkPath& b) {
    const auto aVerbs = a.verbs();
    const auto bVerbs = b.verbs();
    const auto aPoints = a.points();
    const auto bPoints = b.points();
    const auto aConics = a.conicWeights();
    const auto bConics = b.conicWeights();

    return a.getFillType() == b.getFillType() &&
           aVerbs.size() == bVerbs.size() &&
           aPoints.size() == bPoints.size() &&
           aConics.size() == bConics.size() &&
           same_verbs(aVerbs.data(), bVerbs.data(), aVerbs.size()) &&
           same_points(aPoints.data(), bPoints.data(), aPoints.size()) &&
           same_scalars(aConics.data(), bConics.data(), aConics.size());
}

SkPath try_rebuild_analytic_oval(const SkPath& path) {
    if (path.getSegmentMasks() != SkPath::kConic_SegmentMask ||
        path.countVerbs() != 6 ||
        path.conicWeights().size() != 4) {
        return path;
    }

    const SkRect bounds = path.getBounds();
    if (bounds.isEmpty()) {
        return path;
    }

    for (SkPathDirection dir : {SkPathDirection::kCW, SkPathDirection::kCCW}) {
        for (unsigned startIndex = 0; startIndex < 4; ++startIndex) {
            SkPath candidate = SkPath::Oval(bounds, dir, startIndex);
            candidate.setFillType(path.getFillType());
            if (paths_match_exactly(candidate, path)) {
                return candidate;
            }
        }
    }

    return path;
}

SkPath try_rebuild_analytic_rrect(const SkPath& path) {
    const uint32_t rrectMask = SkPath::kLine_SegmentMask | SkPath::kConic_SegmentMask;
    if (path.getSegmentMasks() != rrectMask ||
        path.countVerbs() != 10 ||
        path.conicWeights().size() != 4) {
        return path;
    }

    const SkRect bounds = path.getBounds();
    if (bounds.isEmpty()) {
        return path;
    }

    const SkRRect rrect = SkPathPriv::DeduceRRectFromContour(bounds, path.points(), path.verbs());
    if (!rrect.isValid()) {
        return path;
    }

    for (SkPathDirection dir : {SkPathDirection::kCW, SkPathDirection::kCCW}) {
        for (unsigned startIndex = 0; startIndex < 8; ++startIndex) {
            SkPath candidate = SkPath::RRect(rrect, dir, startIndex);
            candidate.setFillType(path.getFillType());
            if (paths_match_exactly(candidate, path)) {
                return candidate;
            }
        }
    }

    return path;
}

}  // namespace

SkPath RestoreAnalyticSkPath(const SkPath& path) {
    SkPath analytic = try_rebuild_analytic_oval(path);
    analytic = try_rebuild_analytic_rrect(analytic);
    return analytic;
}

void Path::moveTo(float x, float y) {
    fCommands.push_back(Command{Verb::kMove, Vec2{x, y}, {}, {}});
}

void Path::lineTo(float x, float y) {
    fCommands.push_back(Command{Verb::kLine, Vec2{x, y}, {}, {}});
}

void Path::quadTo(float cx, float cy, float x, float y) {
    fCommands.push_back(Command{Verb::kQuad, Vec2{cx, cy}, Vec2{x, y}, {}});
}

void Path::conicTo(float cx, float cy, float x, float y, float w) {
    fCommands.push_back(Command{Verb::kConic, Vec2{cx, cy}, Vec2{x, y}, {}, w});
}

void Path::cubicTo(float cx1, float cy1, float cx2, float cy2, float x, float y) {
    fCommands.push_back(Command{Verb::kCubic, Vec2{cx1, cy1}, Vec2{cx2, cy2}, Vec2{x, y}});
}

void Path::close() {
    fCommands.push_back(Command{Verb::kClose, {}, {}, {}});
}

SkPath ToSkPath(const Path& path) {
    SkPathBuilder builder;
    builder.setFillType(path.inverseFill()
                                ? (path.fillRule() == FillRule::kEvenOdd
                                           ? SkPathFillType::kInverseEvenOdd
                                           : SkPathFillType::kInverseWinding)
                                : (path.fillRule() == FillRule::kEvenOdd
                                           ? SkPathFillType::kEvenOdd
                                           : SkPathFillType::kWinding));
    for (const Path::Command& cmd : path.commands()) {
        switch (cmd.verb) {
            case Path::Verb::kMove:
                builder.moveTo(cmd.p0.x, cmd.p0.y);
                break;
            case Path::Verb::kLine:
                builder.lineTo(cmd.p0.x, cmd.p0.y);
                break;
            case Path::Verb::kQuad:
                builder.quadTo(cmd.p0.x, cmd.p0.y, cmd.p1.x, cmd.p1.y);
                break;
            case Path::Verb::kConic:
                builder.conicTo(cmd.p0.x, cmd.p0.y, cmd.p1.x, cmd.p1.y, cmd.w);
                break;
            case Path::Verb::kCubic:
                builder.cubicTo(cmd.p0.x, cmd.p0.y, cmd.p1.x, cmd.p1.y, cmd.p2.x, cmd.p2.y);
                break;
            case Path::Verb::kClose:
                builder.close();
                break;
        }
    }
    SkPath skPath = builder.detach();
    return RestoreAnalyticSkPath(skPath);
}

bool BuildPathMesh(const Path& path,
                   const MeshBuildOptions& options,
                   MeshData* outMesh,
                   std::string* error) {
    if (error) {
        *error = "Legacy GrTriangulator fill path removed; use CapturePathDrawPlanOriginalSkia instead";
    }
    return false;
}

}  // namespace skia_port
