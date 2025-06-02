#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <iostream>
#include <c10/core/impl/alloc_cpu.h>

namespace foo_core {

// =====================================
// ========= Custom Allocators =========
// =====================================

// PyTorch provides an API for registering custom allocators for your device.
// You can create one by inheriting from the c10::Allocator class,
// and registering your allocator for the particular device type
// (PrivateUse1 for open registration devices)

// A dummy allocator for our custom device, that secretly uses the CPU
struct DummyAllocator final : c10::Allocator {
    DummyAllocator() = default;

    c10::DataPtr allocate(size_t nbytes) override
    {
        std::cout << "Custom allocator's allocate() called!" << std::endl;
        void* data = c10::alloc_cpu(nbytes);
        return {data, data, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1, 0)};
    }

    static void ReportAndDelete(void* ptr)
    {
        if (!ptr) {
            return;
        }
        std::cout << "Custom allocator's delete() called!" << std::endl;
        c10::free_cpu(ptr);
    }

    c10::DeleterFnPtr raw_deleter() const override
    {
        return &ReportAndDelete;
    }

    void copy_data(void* dest, const void* src, std::size_t count) const final
    {
        default_copy_data(dest, src, count);
    }
};

// Register the allocator
static DummyAllocator global_dummy_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_dummy_alloc);

} // namespace foo_core
