#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <vector>
#include <string>

struct Buffer {
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    size_t size;
};

struct ComputeProgram {
    vk::ShaderModule shaderModule;
    vk::DescriptorSetLayout descriptorLayout;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
    vk::DescriptorPool descriptorPool;
    vk::DescriptorSet descriptorSet;
};

// Holds instance, physical device, logical device, queue and command pool.
class VulkanContext {
public:
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::Queue computeQueue;
    uint32_t queueFamilyIndex;
    vk::CommandPool commandPool;

    VulkanContext();
    ~VulkanContext();
};

// Creates a host‑visible storage buffer for read‑only or read‑write access.
Buffer createBuffer(VulkanContext& ctx, size_t count, size_t dtypeSize, bool readOnly);

// Releases the buffer and its memory.
void destroyBuffer(VulkanContext& ctx, Buffer& buf);

// Compiles a GLSL compute shader and builds a compute pipeline with descriptors.
ComputeProgram createComputeProgramFromGLSL(VulkanContext& ctx,
    const std::string& glsl_source,
    const std::vector<Buffer*>& readonlyBuffers,
    const std::vector<Buffer*>& readwriteBuffers);

// Destroys the compute program and associated resources.
void destroyComputeProgram(VulkanContext& ctx, ComputeProgram& prog);

// Dispatches a compute program with the given number of invocations.
void runProgram(VulkanContext& ctx, ComputeProgram& prog, uint32_t numInvocations);