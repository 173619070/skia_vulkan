#include "include/core/SkData.h"

#include "include/core/SkStream.h"
#include "include/private/base/SkAssert.h"
#include "include/private/base/SkMalloc.h"

#include <cstdio>
#include <cstring>
#include <new>

#if defined(_WIN32)
#include <io.h>
#endif

SkData::SkData(SkSpan<std::byte> span, ReleaseProc proc, void* context)
    : fReleaseProc(proc)
    , fReleaseProcContext(context)
    , fSpan(span) {}

SkData::SkData(size_t size)
    : fReleaseProc(nullptr)
    , fReleaseProcContext(nullptr)
    , fSpan{reinterpret_cast<std::byte*>(this + 1), size} {}

SkData::~SkData() {
    if (fReleaseProc) {
        fReleaseProc(fSpan.data(), fReleaseProcContext);
    }
}

bool SkData::operator==(const SkData& other) const {
    if (this == &other) {
        return true;
    }
    return this->size() == other.size() && !sk_careful_memcmp(this->data(), other.data(), this->size());
}

size_t SkData::copyRange(size_t offset, size_t length, void* buffer) const {
    size_t available = this->size();
    if (offset >= available || length == 0) {
        return 0;
    }
    available -= offset;
    if (length > available) {
        length = available;
    }
    if (buffer) {
        std::memcpy(buffer, this->bytes() + offset, length);
    }
    return length;
}

sk_sp<SkData> SkData::shareSubset(size_t offset, size_t length) {
    if (offset > this->size() || length > this->size() - offset) {
        return nullptr;
    }
    if (offset == 0 && length == this->size()) {
        return sk_ref_sp(this);
    }
    if (length == 0) {
        return SkData::MakeEmpty();
    }

    this->ref();
    return SkData::MakeWithProc(this->bytes() + offset, length, [](const void*, void* ctx) {
        static_cast<SkData*>(ctx)->unref();
    }, this);
}

sk_sp<const SkData> SkData::shareSubset(size_t offset, size_t length) const {
    return const_cast<SkData*>(this)->shareSubset(offset, length);
}

sk_sp<SkData> SkData::copySubset(size_t offset, size_t length) const {
    if (offset > this->size() || length > this->size() - offset) {
        return nullptr;
    }
    return SkData::MakeWithCopy(this->bytes() + offset, length);
}

void SkData::operator delete(void* p) {
    ::operator delete(p);
}

sk_sp<SkData> SkData::PrivateNewWithCopy(const void* srcOrNull, size_t length) {
    if (length == 0) {
        return SkData::MakeEmpty();
    }

    const size_t actualLength = length + sizeof(SkData);
    SkASSERT_RELEASE(length < actualLength);

    void* storage = ::operator new(actualLength);
    sk_sp<SkData> data(new (storage) SkData(length));
    if (srcOrNull) {
        std::memcpy(data->writable_data(), srcOrNull, length);
    }
    return data;
}

void SkData::NoopReleaseProc(const void*, void*) {}

sk_sp<SkData> SkData::MakeEmpty() {
    static SkData* empty = new SkData({}, nullptr, nullptr);
    return sk_ref_sp(empty);
}

namespace {

void sk_free_releaseproc(const void* ptr, void*) {
    sk_free(const_cast<void*>(ptr));
}

void stdio_releaseproc(const void* ptr, void*) {
    std::free(const_cast<void*>(ptr));
}

sk_sp<SkData> make_data_from_owned_file(FILE* f) {
    if (!f) {
        return nullptr;
    }
    auto data = SkData::MakeFromFILE(f);
    std::fclose(f);
    return data;
}

bool measure_remaining_file_size(FILE* f, size_t* size) {
    const auto original = std::ftell(f);
    if (original < 0 || std::fseek(f, 0, SEEK_END) != 0) {
        return false;
    }

    const auto end = std::ftell(f);
    if (end < 0 || std::fseek(f, original, SEEK_SET) != 0) {
        return false;
    }

    *size = static_cast<size_t>(end - original);
    return true;
}

bool read_file_bytes(FILE* f, void* storage, size_t size) {
    return std::fread(storage, 1, size, f) == size;
}

FILE* open_read_only_duplicate_file(int fd) {
#if defined(_WIN32)
    int dupFd = _dup(fd);
    if (dupFd < 0) {
        return nullptr;
    }
    return _fdopen(dupFd, "rb");
#else
    int dupFd = dup(fd);
    if (dupFd < 0) {
        return nullptr;
    }
    return fdopen(dupFd, "rb");
#endif
}

}  // namespace

sk_sp<SkData> SkData::MakeFromMalloc(const void* data, size_t length) {
    auto* ptr = static_cast<std::byte*>(const_cast<void*>(data));
    return sk_sp<SkData>(new SkData({ptr, length}, sk_free_releaseproc, nullptr));
}

sk_sp<SkData> SkData::MakeWithCopy(const void* src, size_t length) {
    SkASSERT(src || length == 0);
    return PrivateNewWithCopy(src, length);
}

sk_sp<SkData> SkData::MakeUninitialized(size_t length) {
    return PrivateNewWithCopy(nullptr, length);
}

sk_sp<SkData> SkData::MakeZeroInitialized(size_t length) {
    auto data = MakeUninitialized(length);
    if (length != 0) {
        std::memset(data->writable_data(), 0, data->size());
    }
    return data;
}

sk_sp<SkData> SkData::MakeWithProc(const void* data, size_t length, ReleaseProc proc, void* ctx) {
    auto* ptr = static_cast<std::byte*>(const_cast<void*>(data));
    return sk_sp<SkData>(new SkData({ptr, length}, proc, ctx));
}

sk_sp<SkData> SkData::MakeFromFILE(FILE* f) {
    if (!f) {
        return nullptr;
    }

    size_t size = 0;
    if (!measure_remaining_file_size(f, &size)) {
        return nullptr;
    }
    if (size == 0) {
        return SkData::MakeEmpty();
    }

    void* storage = std::malloc(size);
    if (!storage) {
        return nullptr;
    }

    if (!read_file_bytes(f, storage, size)) {
        std::free(storage);
        return nullptr;
    }

    return SkData::MakeWithProc(storage, size, stdio_releaseproc, nullptr);
}

sk_sp<SkData> SkData::MakeFromFileName(const char path[]) {
    return make_data_from_owned_file(path ? std::fopen(path, "rb") : nullptr);
}

sk_sp<SkData> SkData::MakeFromFD(int fd) {
    return make_data_from_owned_file(open_read_only_duplicate_file(fd));
}

sk_sp<SkData> SkData::MakeWithCString(const char cstr[]) {
    if (!cstr) {
        cstr = "";
    }
    return MakeWithCopy(cstr, std::strlen(cstr) + 1);
}

sk_sp<SkData> SkData::MakeFromStream(SkStream* stream, size_t size) {
    if (!stream) {
        return nullptr;
    }

    auto data = SkData::MakeUninitialized(size);
    if (size != 0 && stream->read(data->writable_data(), size) != size) {
        return nullptr;
    }
    return data;
}
