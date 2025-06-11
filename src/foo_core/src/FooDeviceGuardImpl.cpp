#include "FooDeviceGuardImpl.h"
#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/DeviceType.h>

namespace foo_core {

static uint16_t CURR_DEVICE = -1;

FooDeviceGuardImpl::FooDeviceGuardImpl(c10::DeviceType t)
{
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1);
}

c10::Device FooDeviceGuardImpl::exchangeDevice(c10::Device d) const
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

c10::Device FooDeviceGuardImpl::getDevice() const
{
    return c10::Device(c10::DeviceType::PrivateUse1, CURR_DEVICE);
}

void FooDeviceGuardImpl::setDevice(c10::Device d) const
{
    TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1);
    TORCH_INTERNAL_ASSERT(d.index() < deviceCount(), "Error: device index ", d.index(), " does not exist.");
    c10::Device current_device = getDevice();
    if (current_device != d) {
        CURR_DEVICE = d.index();
    }
}

void FooDeviceGuardImpl::uncheckedSetDevice(c10::Device d) const noexcept
{
    auto current_device = getDevice();
    if (current_device != d) {
        CURR_DEVICE = d.index();
    }
}

c10::Stream FooDeviceGuardImpl::getStream(c10::Device d) const noexcept
{
    // no-op
    return c10::Stream(c10::Stream::DEFAULT, d);
}

c10::Stream FooDeviceGuardImpl::exchangeStream(c10::Stream) const noexcept
{
    // no-op
    return c10::Stream(c10::Stream::DEFAULT, c10::Device(c10::DeviceType::PrivateUse1, CURR_DEVICE));
}

c10::DeviceIndex FooDeviceGuardImpl::deviceCount() const noexcept
{
    // Hardcoding the number of "valid" devices here at 2.
    return 2;
}

// Event-related functions, determine if need to actually support
void FooDeviceGuardImpl::record(
    void** /*event*/,
    const at::Stream& /*stream*/,
    const at::DeviceIndex /*device_index*/,
    const c10::EventFlag /*flag*/) const
{
    TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.");
}

void FooDeviceGuardImpl::block(void* /*event*/, const at::Stream& /*stream*/) const
{
    TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.")
}

bool FooDeviceGuardImpl::queryEvent(void* /*event*/) const
{
    TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.")
}
void FooDeviceGuardImpl::destroyEvent(void* /*event*/, const at::DeviceIndex /*device_index*/) const noexcept  {}

bool FooDeviceGuardImpl::queryStream(const at::Stream& /*stream*/) const
{
    return true;
}
void FooDeviceGuardImpl::synchronizeStream(const at::Stream& /*stream*/) const
{
    // Don't wait for anything.
}

// Register our DeviceGuardImpl so kernels can use it
C10_REGISTER_GUARD_IMPL(PrivateUse1, FooDeviceGuardImpl);

} // namespace foo_core
