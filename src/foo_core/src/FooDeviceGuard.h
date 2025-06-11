#include "FooDeviceGuardImpl.h"
#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>

namespace foo_core {

// A variant of c10::DeviceGuard specialized for our FooDeviceGuardImpl
// It accepts integer indices (interpreting them as dummy devices) and is
// a little more efficient than DeviceGuard (it compiles to straight line
// FooDeviceGuard::[set|get]Device calls). However, it can only be used
// from code that links against this directly.
// It also provides a good place to put documentation and a way to avoid
// explosing nasty template errors.
struct FooDeviceGuard {
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
    explicit FooDeviceGuard() = delete;

    /// Set the current dummy device to the passed device index
    explicit FooDeviceGuard(c10::DeviceIndex device_index) : guard_(device_index) {}

    /// Set the current dummy device to the passed device. Errors if the passed device
    /// is not a dummy device.
    explicit FooDeviceGuard(c10::Device device) : guard_(device) {}

    // Copy is not allowed
    FooDeviceGuard(const FooDeviceGuard&) = delete;
    FooDeviceGuard& operator=(const FooDeviceGuard&) = delete;

    // Move is disallowed, as device guards do not have uninitialized state,
    // which is required for moves on types with nontrivial destructors
    FooDeviceGuard(FooDeviceGuard&& other) = delete;
    FooDeviceGuard& operator=(FooDeviceGuard&& other) = delete;

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
    c10::impl::InlineDeviceGuard<FooDeviceGuardImpl> guard_;
};
}
