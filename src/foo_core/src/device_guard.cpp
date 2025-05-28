#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/DeviceType.h>

C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::NoOpDeviceGuardImpl<at::DeviceType::PrivateUse1>);
