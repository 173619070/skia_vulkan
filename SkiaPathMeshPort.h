#pragma once

#include <ported_skia/include/core/SkPath.h>

#include <cstdint>
#include <string>
#include <vector>

namespace skia_port {

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

enum class FillRule {
    kWinding,
    kEvenOdd,
};

class Path {
public:
    enum class Verb {
        kMove,
        kLine,
        kQuad,
        kConic,
        kCubic,
        kClose,
    };

    struct Command {
        Verb verb = Verb::kMove;
        Vec2 p0{};
        Vec2 p1{};
        Vec2 p2{};
        float w = 1.0f;
    };

    void setFillRule(FillRule rule) { fFillRule = rule; }
    void setInverseFill(bool inverse) { fInverseFill = inverse; }
    FillRule fillRule() const { return fFillRule; }
    bool inverseFill() const { return fInverseFill; }

    void moveTo(float x, float y);
    void lineTo(float x, float y);
    void quadTo(float cx, float cy, float x, float y);
    void conicTo(float cx, float cy, float x, float y, float w);
    void cubicTo(float cx1, float cy1, float cx2, float cy2, float x, float y);
    void close();

    const std::vector<Command>& commands() const { return fCommands; }

private:
    FillRule fFillRule = FillRule::kWinding;
    bool fInverseFill = false;
    std::vector<Command> fCommands;
};

// Legacy adapter for non-main-chain code paths. Phase 1 moves the tessellation main chain to
// consume SkPath directly; this helper remains for debug/demo code that still builds skia_port::Path.
SkPath ToSkPath(const Path& path);
SkPath RestoreAnalyticSkPath(const SkPath& path);

struct MeshBuildOptions {
    // Equivalent idea to Skia's default tolerance in GrPathUtils.
    float flattenTolerance = 0.25f;
    float simplifyEpsilon = 1e-4f;
    bool autoCloseOpenContours = true;
    bool requireSimpleContours = true;
};

enum class StrokeCap {
    kButt,
    kRound,
    kSquare
};

enum class StrokeJoin {
    kMiter,
    kRound,
    kBevel
};

struct StrokeOptions {
    float width = 1.0f;
    float miterLimit = 4.0f;
    StrokeCap cap = StrokeCap::kButt;
    StrokeJoin join = StrokeJoin::kMiter;
    std::vector<float> dashArray;
    float dashOffset = 0.0f;
};

struct MeshData {
    std::vector<Vec2> vertices;
    std::vector<uint32_t> indices;
};

struct Mat3 {
    // Row-major 3x3, matching SkMatrix::set9 layout.
    float v[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
};

struct RectF {
    float left = 0.0f;
    float top = 0.0f;
    float right = 0.0f;
    float bottom = 0.0f;
};

struct Triangle {
    Vec2 p0{};
    Vec2 p1{};
    Vec2 p2{};
};

struct PatchPrepareOptions {
    // Path-space -> device approximation used for chop-if-necessary checks.
    Mat3 viewMatrix{};
    // Transform used by PatchWriter::setShaderTransform (Wang estimate in shader space).
    Mat3 shaderMatrix{};
    // Per-path transform applied to control points before writing patches.
    Mat3 pathMatrix{};

    RectF clipConservativeBounds{};
    bool hasClipConservativeBounds = false;

    // Mirrors PathTessellator(bool infinitySupport, ...): if false, force explicit curve type.
    bool infinitySupport = false;
    // Mirrors GrShaderCaps::fVertexIDSupport for fixed topology / hull / bbox paths.
    bool vertexIDSupport = true;
    // Mirrors FillPathFlags::kStencilOnly / kWireframe.
    bool stencilOnly = false;
    bool wireframe = false;
    // Mirrors pipeline()->hasStencilClip() in PathInnerTriangulateOp.
    bool hasStencilClip = false;
    // Mirrors TessellationPathRenderer::ChopPathIfNecessary gate.
    bool preChopCurvesIfNecessary = true;

    // Optional breadcrumb triangles equivalent to prepareWithTriangles(extraTriangles).
    std::vector<Triangle> extraTriangles;
};

enum class AAMode {
    kNone,
    kMSAA,
    // Upstream TessellationPathRenderer rejects coverage AA in onCanDrawPath().
    kCoverage,
};

struct PatchBufferData {
    struct Chunk {
        uint32_t basePatch = 0;
        uint32_t patchCount = 0;
        uint32_t byteOffset = 0;
        uint32_t byteSize = 0;
    };

    std::vector<uint8_t> data;
    std::vector<Chunk> chunks;
    uint32_t patchStrideBytes = 0;
    uint32_t patchCount = 0;
    uint32_t attribMask = 0;
    uint32_t preallocPatchCount = 0;
    int requiredResolveLevel = 0;
    int maxFixedCountVertexCount = 0;

    // Fixed-count static template buffers equivalent to FixedCountCurves/FixedCountWedges.
    std::vector<uint8_t> fixedVertexBufferTemplate;
    std::vector<uint8_t> fixedIndexBufferTemplate;
    uint32_t fixedVertexStrideBytes = 0;
    uint32_t fixedVertexCount = 0;
    uint32_t fixedIndexCount = 0;
};

enum class TessPlanOpKind {
    kUnknown,
    kPathCurveTessellator,
    kPathWedgeTessellator,
    kPathInnerTriangulateOp,
    kPathStencilCoverOp,
    kPathTessellateOp,
    kStrokeTessellateOp,
};

enum class TessPlanPassKind {
    kUnknown,
    kStencilCurvePatches,
    kStencilWedgePatches,
    kStencilFanTriangles,
    kFillFanTriangles,
    kCoverHulls,
    kCoverBoundingBoxes,
    kStrokePatches,
};

enum class TessDrawCommandKind {
    kUnknown,
    kDraw,
    kIndexedInstanced,
    kInstanced,
};

enum class TessPrimitiveType {
    kUnknown,
    kTriangles,
    kTriangleStrip,
    kPatches,
};

enum class TessStencilSettingsKind {
    kUnknown,
    kUnused,
    kStencilPath,
    kTestAndReset,
    kTestAndResetInverse,
    kFillOrIncrDecr,
    kFillOrInvert,
    kFillIfZeroAndInClip,
    kIncrDecrStencilIfNonzero,
    kInvertStencilIfNonzero,
};

enum class TessStencilTestKind {
    kAlwaysIfInClip,
    kEqualIfInClip,
    kLessIfInClip,
    kLEqualIfInClip,
    kAlways,
    kNever,
    kGreater,
    kGEqual,
    kLess,
    kLEqual,
    kEqual,
    kNotEqual,
};

enum class TessStencilOpKind {
    kKeep,
    kZero,
    kReplace,
    kInvert,
    kIncWrap,
    kDecWrap,
    kIncMaybeClamp,
    kDecMaybeClamp,
    kZeroClipBit,
    kSetClipBit,
    kInvertClipBit,
    kSetClipAndReplaceUserBits,
    kZeroClipAndUserBits,
};

struct TessStencilFaceInfo {
    uint16_t ref = 0;
    TessStencilTestKind test = TessStencilTestKind::kAlways;
    uint16_t testMask = 0;
    TessStencilOpKind passOp = TessStencilOpKind::kKeep;
    TessStencilOpKind failOp = TessStencilOpKind::kKeep;
    uint16_t writeMask = 0;
};

struct TessStencilSettingsInfo {
    TessStencilSettingsKind kind = TessStencilSettingsKind::kUnknown;
    bool usesStencil = false;
    bool testsStencil = false;
    bool writesStencil = false;
    bool resetsStencil = false;
    bool usesClipBit = false;
    bool twoSided = false;
    TessStencilFaceInfo front;
    TessStencilFaceInfo back;
};

enum class TessProgramKind {
    kUnknown,
    kStencilFan,
    kStencilPath,
    kCoverBoundingBox,
    kStencilCurves,
    kFanFill,
    kFanStencil,
    kCoverHulls,
};

enum class TessLayoutKind {
    kUnknown,
    kSimpleTriangle,
    kMiddleOut,
    kBoundingBox,
    kHull,
    kStroke,
};

struct TessPrimitiveProcessorLayout {
    TessLayoutKind kind = TessLayoutKind::kUnknown;
    uint32_t vertexStrideBytes = 0;
    uint32_t instanceStrideBytes = 0;
    uint32_t vertexAttributeCount = 0;
    uint32_t instanceAttributeCount = 0;
    bool hasVertexAttributes = false;
    bool hasInstanceAttributes = false;
    bool usesVertexID = true;
};

struct TessStrokeProgramInfo {
    bool enabled = false;
    bool hasDynamicStroke = false;
    bool hairline = false;
    float numRadialSegmentsPerRadian = 0.0f;
    float joinType = 0.0f;
    float strokeRadius = 0.0f;
};

struct TessProgramInfo {
    TessProgramKind kind = TessProgramKind::kUnknown;
    TessPrimitiveType primitiveType = TessPrimitiveType::kUnknown;
    uint32_t numSamples = 1;
    TessStencilSettingsInfo stencilSettings;
    TessPrimitiveProcessorLayout layout;
    uint32_t patchAttribMask = 0;
    bool usesVertexID = true;
    bool hasStencilClip = false;
    bool wireframe = false;
    bool stencilOnly = false;
    TessStrokeProgramInfo stroke;
};

struct TessDrawCommand {
    TessDrawCommandKind kind = TessDrawCommandKind::kUnknown;
    uint32_t elementCount = 0;
    uint32_t baseIndex = 0;
    uint32_t instanceCount = 0;
    uint32_t baseInstance = 0;
    uint32_t baseVertex = 0;
    uint32_t boundIndexBufferBytes = 0;
    uint32_t boundInstanceBufferBytes = 0;
    uint32_t boundVertexBufferBytes = 0;
};

struct TessInstanceBufferData {
    std::vector<uint8_t> data;
    uint32_t strideBytes = 0;
    uint32_t instanceCount = 0;
};

struct TessNodeMetadata {
    std::string nodeTag;
    std::string nodeId;
    uint32_t nodeIndex = 0;
    uint32_t contourIndex = 0;
};

struct TessInputPathInfo {
    uint32_t fillType = 0;
    bool inverseFill = false;
    bool isFinite = false;
    bool isVolatile = false;
    bool isLastContourClosed = false;
    bool isLine = false;
    bool isRect = false;
    bool rectClosed = false;
    uint32_t rectDirection = 0;
    bool isOval = false;
    bool isRRect = false;
    uint32_t rrectType = 0;
    uint32_t segmentMask = 0;
    uint32_t verbCount = 0;
    uint32_t pointCount = 0;
    uint32_t conicWeightCount = 0;
    RectF bounds{};
    RectF tightBounds{};
    RectF rectBounds{};
    RectF ovalBounds{};
    RectF rrectBounds{};
    Vec2 linePoints[2]{};
    Vec2 rrectRadii[4]{};
};

struct TessNodeStyleInfo {
    FillRule fillRule = FillRule::kWinding;
    float strokeWidth = 0.0f;
    float strokeMiterLimit = 4.0f;
    StrokeCap strokeCap = StrokeCap::kButt;
    StrokeJoin strokeJoin = StrokeJoin::kMiter;
    float dashOffset = 0.0f;
    std::vector<float> dashArray;
};

struct TessNodePaintInfo {
    Mat3 nodeTransform{};
    float currentColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float resolvedColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float opacity = 1.0f;
    float fillOpacity = 1.0f;
    float strokeOpacity = 1.0f;
};

struct TessPassPlan {
    TessPlanPassKind kind = TessPlanPassKind::kUnknown;
    TessProgramInfo programInfo;
    Mat3 shaderMatrix{};
    PatchBufferData patchBuffer;
    MeshData triangleMesh;
    TessInstanceBufferData instanceBuffer;
    std::vector<TessDrawCommand> drawCommands;
    std::string debugName;
};

struct TessCapturePlan {
    TessPlanOpKind opKind = TessPlanOpKind::kUnknown;
    bool usedOriginalSkiaCore = false;
    bool complete = false;
    TessNodeMetadata node;
    TessInputPathInfo inputPath;
    TessNodeStyleInfo nodeStyle;
    TessNodePaintInfo nodePaint;
    std::vector<TessPassPlan> passes;
    float color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
};

// Pipeline:
//   path commands -> flatten Bezier curves -> simplify contour points -> triangulate
// Result is a vertex/index triangle list (no rendering dependency).
bool BuildPathMesh(const Path& path,
                   const MeshBuildOptions& options,
                   MeshData* outMesh,
                   std::string* error = nullptr);

// Uses source-ported Skia PatchWriter to emit patch instances (CPU-side raw patch buffer).
bool BuildPathPatchBufferCopiedSkia(const SkPath& path,
                                    const MeshBuildOptions& options,
                                    PatchBufferData* outPatches,
                                    std::string* error = nullptr);

bool BuildPathPatchBufferCopiedSkia(const Path& path,
                                    const MeshBuildOptions& options,
                                    PatchBufferData* outPatches,
                                    std::string* error = nullptr);

// Extended prepare path that mirrors PathTessellator::prepare/prepareWithTriangles data flow.
bool BuildPathPatchBufferCopiedSkia(const SkPath& path,
                                    const MeshBuildOptions& options,
                                    const PatchPrepareOptions& prepareOptions,
                                    PatchBufferData* outPatches,
                                    std::string* error = nullptr);

bool BuildPathPatchBufferCopiedSkia(const Path& path,
                                    const MeshBuildOptions& options,
                                    const PatchPrepareOptions& prepareOptions,
                                    PatchBufferData* outPatches,
                                    std::string* error = nullptr);

// Independent reference generator that mirrors PathCurveTessellator::prepare dataflow
// (prepareWithTriangles + write_curve_patches + fixed-count template setup).
bool BuildPathPatchBufferReferenceSkiaNativeFlow(const SkPath& path,
                                                 const MeshBuildOptions& options,
                                                 const PatchPrepareOptions& prepareOptions,
                                                 PatchBufferData* outPatches,
                                                 std::string* error = nullptr);

bool BuildPathPatchBufferReferenceSkiaNativeFlow(const Path& path,
                                                 const MeshBuildOptions& options,
                                                 const PatchPrepareOptions& prepareOptions,
                                                 PatchBufferData* outPatches,
                                                 std::string* error = nullptr);

// Legacy compatibility wrappers around the formal draw-plan capture path.
// Phase 6 routes these APIs through the same runtime capture used by
// CapturePathDrawPlanOriginalSkia(), so pass metadata now comes from
// finalize/onPrepare/onExecute instead of CpuPrePrepare* inference.
bool CapturePathTessellationPlanOriginalSkia(const SkPath& path,
                                             const PatchPrepareOptions& prepareOptions,
                                             TessCapturePlan* outPlan,
                                             std::string* error = nullptr);

bool CapturePathTessellationPlanOriginalSkia(const Path& path,
                                             const PatchPrepareOptions& prepareOptions,
                                             TessCapturePlan* outPlan,
                                             std::string* error = nullptr);

bool CaptureStrokeTessellationPlanOriginalSkia(const SkPath& path,
                                               const StrokeOptions& strokeOptions,
                                               const PatchPrepareOptions& prepareOptions,
                                               TessCapturePlan* outPlan,
                                               std::string* error = nullptr);

bool CaptureStrokeTessellationPlanOriginalSkia(const Path& path,
                                               const StrokeOptions& strokeOptions,
                                               const PatchPrepareOptions& prepareOptions,
                                               TessCapturePlan* outPlan,
                                               std::string* error = nullptr);

struct PathDrawOptions {
    bool isStroke = false;
    StrokeOptions strokeOptions;
    PatchPrepareOptions patchOptions;
    AAMode aaType = AAMode::kNone;
};

// P9: Top Level Draw Plan Router mimicking TessellationPathRenderer::onDrawPath
bool CapturePathDrawPlanOriginalSkia(const SkPath& path,
                                     const PathDrawOptions& options,
                                     TessCapturePlan* outPlan,
                                     std::string* error = nullptr);

bool CapturePathDrawPlanOriginalSkia(const Path& path,
                                     const PathDrawOptions& options,
                                     TessCapturePlan* outPlan,
                                     std::string* error = nullptr);

}  // namespace skia_port
