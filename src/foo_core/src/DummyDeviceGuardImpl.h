#pragma once


#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
namespace foo_core {

// =====================================
// ============= Device Guards =========
// =====================================

// PyTorch has an API for registering device guards.
// Device guards can be used to set the current "active" device,
// and e.g. error if the user provides an invalid device index.
//
// If your device doesn't support indices (e.g. foo:0 vs. foo:1),
// then the guards probably aren't needed.
//
// You can use it by creating a DeviceGuard class, registering it
// in PyTorch, and invoking the device guard before any kernels are called.
// For a more full-featured example of a device guard,
// check out the code at c10/cuda/CUDAGuard.h

// Represents the current "active" device.
// The dummy device guard registered below is meant to show how a backend
// can integrate custom device guard with pytorch.
// For something like cuda this represents the current active cuda device,
// which is directly set using the cuda API calls cudaGetDevice/cudaSetDevice.

// Design note: In principle, pytorch could only use classes that extend DeviceGuardImpl
// directly, meaning that DeviceGuard is just a pretty wrapper / boilerplate.
// This is techinically true, but doing it this way allows pytorch better error messages
// and easier specializations (e.g. return CUDAStream instead of Stream inside CUDAStreamGuard)
// DeviceGuardImpl is what will actually get registered, and DeviceGuard is what is used
// inside the code internals like kernels. See the note at the beginning of InlineDeviceGuard.h
//
// TODO: put this in an impl namespace.
struct DummyDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

    DummyDeviceGuardImpl() {}
    explicit DummyDeviceGuardImpl(c10::DeviceType t);
    c10::DeviceType type() const override
    {
        return c10::DeviceType::PrivateUse1;
    }
    c10::Device exchangeDevice(c10::Device d) const override;
    c10::Device getDevice() const override;
    void setDevice(c10::Device d) const override;
    void uncheckedSetDevice(c10::Device d) const noexcept override;

    c10::Stream getStream(c10::Device d) const noexcept override;

    // NB: These do NOT set the current device
    c10::Stream exchangeStream(c10::Stream s) const noexcept override;
    c10::DeviceIndex deviceCount() const noexcept override;
    bool queryStream(const c10::Stream& strema) const override;
    void synchronizeStream(const c10::Stream& stream) const override;

    // Event-related functions
    void record(void** event, const c10::Stream &stream, const c10::DeviceIndex device_index,
                const c10::EventFlag flag) const override;
    void block(void* event, const c10::Stream &stream) const override;
    void destroyEvent(void* event, const c10::DeviceIndex device_index) const noexcept override;
    // Stream-related functions
    bool queryEvent(void *event) const override;
};

} // foo_core
