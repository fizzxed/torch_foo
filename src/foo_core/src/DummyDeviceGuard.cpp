#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/DeviceType.h>

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

// Design note: In principle, pytorch could use classes that extend DeviceGuardImpl
// directly, meaning that DeviceGuard is just a pretty wrapper / boilerplate.
// This is techinically true, but doing it this way allows pytorch better error messages
// and easier specializations (e.g. return CUDAStream instead of Stream inside CUDAStreamGuard)
// DeviceGuardImpl is what will actually get registered, and DeviceGuard is what is used
// inside the code internals like kernels. See the note at the beginning of InlineDeviceGuard.h

// Normally we would seperate DummyGuard and DummyGuardImpl into seperate source files.
//
static uint16_t CURR_DEVICE = -1;

struct DummyDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;
    DummyDeviceGuardImpl() {}

    explicit DummyDeviceGuardImpl(c10::DeviceType t)
    {
        TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1);
    }

    c10::DeviceType type() const override
    {
        return c10::DeviceType::PrivateUse1;
    }

    c10::Device exchangeDevice(c10::Device d) const override
    {
        TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1);
        TORCH_INTERNAL_ASSERT(d.index() < deviceCount(), "Error: device index ", d.index(), " does not exist");
        c10::Device old_device = getDevice();
        if (old_device.index() != d.index()) {
            // set the active device
            CURR_DEVICE = d.index();
        }
        return old_device;
    }

    c10::Device getDevice() const override
    {
        return c10::Device(c10::DeviceType::PrivateUse1, CURR_DEVICE);
    }

    void setDevice(c10::Device d) const override
    {
        TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1);
        TORCH_INTERNAL_ASSERT(d.index() < deviceCount(), "Error: device index ", d.index(), " does not exist.");
        c10::Device current_device = getDevice();
        if (current_device != d) {
            CURR_DEVICE = d.index();
        }
    }

    void uncheckedSetDevice(c10::Device d) const noexcept override
    {
        auto current_device = getDevice();
        if (current_device != d) {
            CURR_DEVICE = d.index();
        }
    }

    c10::Stream getStream(c10::Device d) const noexcept override
    {
        // no-op
        return c10::Stream(c10::Stream::DEFAULT, d);
    }

    // NB: These do NOT set the current device
    c10::Stream exchangeStream(c10::Stream) const noexcept override
    {
        // no-op
        return c10::Stream(c10::Stream::DEFAULT, c10::Device(c10::DeviceType::PrivateUse1, CURR_DEVICE));
    }

    c10::DeviceIndex deviceCount() const noexcept override
    {
        // Hardcoding the number of "valid" devices here at 2.
        return 2;
    }

    // Event-related functions, determine if need to actually support
    void record(
        void** /*event*/,
        const at::Stream& /*stream*/,
        const at::DeviceIndex /*device_index*/,
        const c10::EventFlag /*flag*/) const override
    {
        TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.");
    }
    void block(void* /*event*/, const at::Stream& /*stream*/) const override
    {
        TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.")
    }
    bool queryEvent(void* /*event*/) const override
    {
        TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.")
    }
    void destroyEvent(void* /*event*/, const at::DeviceIndex /*device_index*/) const noexcept override {}

    // Stream-related functions
    bool queryStream(const at::Stream& /*stream*/) const override
    {
        return true;
    }
    void synchronizeStream(const at::Stream& /*stream*/) const override
    {
        // Don't wait for anything.
    }
};

// A variant of c10::DeviceGuard specialized for our DummyDeviceGuardImpl
// It accepts integer indices (interpreting them as dummy devices) and is
// a little more efficient than DeviceGuard (it compiles to straight line
// DummyDeviceGuard::[set|get]Device calls). However, it can only be used
// from code that links against this directly.
// It also provides a good place to put documentation and a way to avoid
// explosing nasty template errors.
struct DummyDeviceGuard {
    /// No default constructor
    /// Note taken from InlineDeviceGuard.h
    /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /// In principle, we could add a default constructor to
    /// DeviceGuard which reads the current device and promises to
    /// restore to that device on exit.  However, most cases where you
    /// would have written this, you probably meant to actually just
    /// use DeviceGuard (since you don't actually need the
    /// restore to happen if you don't ever actually set the device).
    /// We remove the constructor here to encourage you to think about
    /// what you actually want to happen.
    explicit DummyDeviceGuard() = delete;

    /// Set the current dummy device to the passed device index
    explicit DummyDeviceGuard(c10::DeviceIndex device_index) : guard_(device_index) {}

    /// Set the current dummy device to the passed device. Errors if the passed device
    /// is not a dummy device.
    explicit DummyDeviceGuard(c10::Device device) : guard_(device) {}

    // Copy is not allowed
    DummyDeviceGuard(const DummyDeviceGuard&) = delete;
    DummyDeviceGuard& operator=(const DummyDeviceGuard&) = delete;

    // Move is disallowed, as device guards do not have uninitialized state,
    // which is required for moves on types with nontrivial destructors
    DummyDeviceGuard(DummyDeviceGuard&& other) = delete;
    DummyDeviceGuard& operator=(DummyDeviceGuard&& other) = delete;

    /// Sets the dummy device to the given device. Errors if the given device
    /// is not a dummy device
    void set_device(c10::Device device)
    {
        guard_.set_device(device);
    }

    /// Resets the currently set device to its original device, and then sets the
    /// current device to the passed device. This is effectively equivalent to
    /// set_device when a guard supports only a single device type
    void reset_device(c10::Device device)
    {
        guard_.reset_device(device);
    }

    /// Sets the dummy device to the given device index
    void set_index(c10::DeviceIndex device_index)
    {
        guard_.set_index(device_index);
    }

    /// Returns the device that was set upon construction of the guard
    c10::Device original_device() const
    {
        return guard_.original_device();
    }

    /// Returns the last device that was set via `set_device`, if any, otherwise
    /// the device passed during construction.
    c10::Device current_device() const
    {
        return guard_.current_device();
    }

private:
    c10::impl::InlineDeviceGuard<DummyDeviceGuardImpl> guard_;
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, DummyDeviceGuardImpl);
} // namespace foo_core
