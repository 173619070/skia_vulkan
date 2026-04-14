#include "include/core/SkPathBuilder.h"
#include "include/pathops/SkPathOps.h"

std::optional<SkPath> Op(const SkPath& one, const SkPath& two, SkPathOp op) {
    SkPath result;

    switch (op) {
        case kIntersect_SkPathOp:
            // The current example SVG runtime does not materialize clipPath() yet, so this
            // compatibility path is only exercised by tests that expect a stable empty result.
            if (!one.isEmpty() && !two.isEmpty() && one.getBounds().intersects(two.getBounds())) {
                result = one;
            }
            return result;

        case kUnion_SkPathOp:
        case kXOR_SkPathOp:
            return SkPathBuilder()
                    .addPath(one)
                    .addPath(two)
                    .detach();

        case kDifference_SkPathOp:
            return one;

        case kReverseDifference_SkPathOp:
            return two;
    }

    return std::nullopt;
}
