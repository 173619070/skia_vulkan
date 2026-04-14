#pragma once

#include <ported_skia/src/gpu/ganesh/GrGpuBuffer.h>
#include <ported_skia/src/gpu/ganesh/GrResourceProvider.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace skia_port {

struct UniqueKeyHash {
    size_t operator()(const skgpu::UniqueKey& key) const noexcept { return key.hash(); }
};

class CpuGpuBuffer final : public GrGpuBuffer {
public:
    CpuGpuBuffer(size_t sizeInBytes, GrGpuBufferType intendedType, GrAccessPattern accessPattern)
            : GrGpuBuffer(sizeInBytes, intendedType, accessPattern)
            , fData(sizeInBytes)
            , fAllocatedSizeInBytes(sizeInBytes) {}

    uint8_t* writableData() { return fData.data(); }
    const uint8_t* data() const { return fData.data(); }
    size_t allocatedSizeInBytes() const { return fAllocatedSizeInBytes; }

    void resize(size_t sizeInBytes) {
        fData.resize(sizeInBytes);
        fAllocatedSizeInBytes = std::max(fAllocatedSizeInBytes, sizeInBytes);
        this->setSize(sizeInBytes);
    }

private:
    std::vector<uint8_t> fData;
    size_t fAllocatedSizeInBytes = 0;
};

class CpuResourceProvider final : public GrResourceProvider {
public:
    uint32_t contextUniqueID() const override { return 1; }

    sk_sp<const GrGpuBuffer> findOrMakeStaticBuffer(GrGpuBufferType intendedType,
                                                    size_t size,
                                                    const skgpu::UniqueKey& key,
                                                    InitializeBufferFn init) override {
        auto found = fStaticBuffers.find(key);
        if (found != fStaticBuffers.end()) {
            return found->second;
        }

        auto buffer = sk_make_sp<CpuGpuBuffer>(size, intendedType, kStatic_GrAccessPattern);
        if (init && size > 0) {
            skgpu::VertexWriter writer(buffer->writableData(), size);
            init(std::move(writer), size);
        }
        fStaticBuffers.emplace(key, buffer);
        return buffer;
    }

    sk_sp<const GrGpuBuffer> findOrMakeStaticBuffer(GrGpuBufferType intendedType,
                                                    size_t size,
                                                    const void* staticData,
                                                    const skgpu::UniqueKey& key) override {
        auto found = fStaticBuffers.find(key);
        if (found != fStaticBuffers.end()) {
            return found->second;
        }

        auto buffer = sk_make_sp<CpuGpuBuffer>(size, intendedType, kStatic_GrAccessPattern);
        if (staticData && size > 0) {
            std::memcpy(buffer->writableData(), staticData, size);
        }
        fStaticBuffers.emplace(key, buffer);
        return buffer;
    }

private:
    std::unordered_map<skgpu::UniqueKey, sk_sp<const GrGpuBuffer>, UniqueKeyHash> fStaticBuffers;
};

}  // namespace skia_port
