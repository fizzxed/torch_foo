#include <ATen/Context.h> // delete soon
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/StorageImpl.h>
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
struct FooAllocator final : c10::Allocator {
    FooAllocator() = default;

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


c10::intrusive_ptr<c10::StorageImpl> make_custom_storage_impl(
    c10::StorageImpl::use_byte_size_t,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable)
{
    c10::intrusive_ptr<c10::StorageImpl> custom_storage_impl;
    std::cout << "make_storage_impl, bytes = "<< size_bytes << std::endl;
    if (data_ptr == nullptr){
        custom_storage_impl = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), size_bytes, allocator, resizable);
    } else {
        custom_storage_impl = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), size_bytes, std::move(data_ptr), allocator, resizable);
    }
    return custom_storage_impl;
}


// Register the allocator
// static FooAllocator global_foo_alloc;
// REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_foo_alloc);
// Use the CPU Allocator
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, at::getCPUAllocator())

// int register_storage() {
//     std::cout << "Registered storage" << std::endl;
//     c10::SetStorageImplCreate(c10::DeviceType::PrivateUse1, &make_custom_storage_impl);
//     return 0;
// }
// static const int _temp_ = register_storage();



} // namespace foo_core
