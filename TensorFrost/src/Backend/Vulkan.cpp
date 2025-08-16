#include "Backend/Vulkan.h"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#include <shaderc/shaderc.hpp>
#include <stdexcept>

// compile GLSL to SPIR-V at runtime
static std::vector<uint32_t> compileGLSLToSpirv(const std::string& source) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions opts;
    opts.SetTargetEnvironment(shaderc_target_env_vulkan,
                              shaderc_env_version_vulkan_1_1);
    shaderc::SpvCompilationResult result =
        compiler.CompileGlslToSpv(source, shaderc_compute_shader, "shader", opts);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error(result.GetErrorMessage());
    }
    return {result.cbegin(), result.cend()};
}

// VulkanContext constructor sets up instance, selects a compute device and queue, and creates a command pool.
VulkanContext::VulkanContext() {
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);   // required before vk::createInstance

    // 1) instance
    vk::ApplicationInfo appInfo("ComputeFramework", 1, nullptr, 0, VK_API_VERSION_1_1);
    vk::InstanceCreateInfo instCreate({}, &appInfo);
    instance = vk::createInstance(instCreate);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);   // load instance-level funcs

    // 2) pick physical device + compute queue family
    auto devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) throw std::runtime_error("No physical devices");
    for (auto& pd : devices) {
        auto q = pd.getQueueFamilyProperties();
        for (uint32_t i = 0; i < q.size(); ++i) {
            if ( (q[i].queueFlags & vk::QueueFlagBits::eCompute) != vk::QueueFlags{} ) {
                physicalDevice = pd;
                queueFamilyIndex = i;
                break;
            }
        }
        if (physicalDevice) break;
    }
    if (!physicalDevice) throw std::runtime_error("No compute queue");

    // 3) device + queue
    float prio = 1.0f;
    vk::DeviceQueueCreateInfo qci({}, queueFamilyIndex, 1, &prio);
    vk::DeviceCreateInfo devCreate({}, qci);
    device = physicalDevice.createDevice(devCreate);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);     // load device-level funcs

    computeQueue = device.getQueue(queueFamilyIndex, 0);

    // 4) command pool
    vk::CommandPoolCreateInfo poolInfo({}, queueFamilyIndex);
    commandPool = device.createCommandPool(poolInfo);
}

// VulkanContext destructor cleans up the command pool, device and instance.
VulkanContext::~VulkanContext() {
    device.destroyCommandPool(commandPool);
    device.destroy();
    instance.destroy();
}

// create a storage buffer
Buffer createBuffer(VulkanContext& ctx, size_t count, size_t dtypeSize, bool readOnly) {
    Buffer buf;
    buf.size = count * dtypeSize;
    vk::BufferCreateInfo bci({}, buf.size,
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc |
        vk::BufferUsageFlagBits::eTransferDst);
    buf.buffer = ctx.device.createBuffer(bci);
    auto memReq = ctx.device.getBufferMemoryRequirements(buf.buffer);

    auto memProps = ctx.physicalDevice.getMemoryProperties();
    uint32_t memTypeIndex = 0;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        bool allowed = memReq.memoryTypeBits & (1u << i);
        auto typeBits = memReq.memoryTypeBits;
        auto flags    = memProps.memoryTypes[i].propertyFlags;

        bool ok      = (typeBits & (1u << i)) != 0;
        bool hostVis = (flags & vk::MemoryPropertyFlagBits::eHostVisible) != vk::MemoryPropertyFlags{};
        if (allowed && hostVis) {
            memTypeIndex = i;
            break;
        }
    }
    vk::MemoryAllocateInfo allocInfo(memReq.size, memTypeIndex);
    buf.memory = ctx.device.allocateMemory(allocInfo);
    ctx.device.bindBufferMemory(buf.buffer, buf.memory, 0);
    return buf;
}

void destroyBuffer(VulkanContext& ctx, Buffer& buf) {
    ctx.device.destroyBuffer(buf.buffer);
    ctx.device.freeMemory(buf.memory);
    buf.buffer = nullptr;
    buf.memory = nullptr;
}

// internal helper to build a compute program from SPIR-V
static ComputeProgram createComputeProgram(VulkanContext& ctx,
    const std::vector<uint32_t>& spirv,
    const std::vector<Buffer*>& readonlyBuffers,
    const std::vector<Buffer*>& readwriteBuffers) {

    ComputeProgram prog;
    vk::ShaderModuleCreateInfo smci({}, spirv.size() * sizeof(uint32_t), spirv.data());
    prog.shaderModule = ctx.device.createShaderModule(smci);

    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    uint32_t binding = 0;
    for (size_t i = 0; i < readonlyBuffers.size(); i++) {
        bindings.emplace_back(binding++, vk::DescriptorType::eStorageBuffer, 1,
                              vk::ShaderStageFlagBits::eCompute);
    }
    for (size_t i = 0; i < readwriteBuffers.size(); i++) {
        bindings.emplace_back(binding++, vk::DescriptorType::eStorageBuffer, 1,
                              vk::ShaderStageFlagBits::eCompute);
    }
    vk::DescriptorSetLayoutCreateInfo dsInfo({}, bindings.size(), bindings.data());
    prog.descriptorLayout = ctx.device.createDescriptorSetLayout(dsInfo);
    vk::PipelineLayoutCreateInfo plInfo({}, 1, &prog.descriptorLayout);
    prog.pipelineLayout = ctx.device.createPipelineLayout(plInfo);

    vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute,
                                                prog.shaderModule, "main");
    vk::ComputePipelineCreateInfo cpInfo({}, stageInfo, prog.pipelineLayout);
    prog.pipeline = ctx.device.createComputePipeline({}, cpInfo).value;

    vk::DescriptorPoolSize poolSize(vk::DescriptorType::eStorageBuffer,
                                    readonlyBuffers.size() + readwriteBuffers.size());
    vk::DescriptorPoolCreateInfo poolInfo({}, 1, 1, &poolSize);
    prog.descriptorPool = ctx.device.createDescriptorPool(poolInfo);
    vk::DescriptorSetAllocateInfo allocInfo(prog.descriptorPool, 1, &prog.descriptorLayout);
    prog.descriptorSet = ctx.device.allocateDescriptorSets(allocInfo)[0];

    std::vector<vk::DescriptorBufferInfo> bufferInfos;
    bufferInfos.reserve(readonlyBuffers.size() + readwriteBuffers.size());
    for (auto b : readonlyBuffers) {
        bufferInfos.push_back(vk::DescriptorBufferInfo(b->buffer, 0, b->size));
    }
    for (auto b : readwriteBuffers) {
        bufferInfos.push_back(vk::DescriptorBufferInfo(b->buffer, 0, b->size));
    }
    std::vector<vk::WriteDescriptorSet> writes;
    for (uint32_t i = 0; i < bufferInfos.size(); i++) {
        vk::WriteDescriptorSet w(prog.descriptorSet, i, 0, 1,
                                 vk::DescriptorType::eStorageBuffer, nullptr,
                                 &bufferInfos[i]);
        writes.push_back(w);
    }
    ctx.device.updateDescriptorSets(writes, {});
    return prog;
}

// public wrapper that compiles GLSL and builds the program
ComputeProgram createComputeProgramFromGLSL(VulkanContext& ctx,
    const std::string& glsl_source,
    const std::vector<Buffer*>& readonlyBuffers,
    const std::vector<Buffer*>& readwriteBuffers) {

    auto spirv = compileGLSLToSpirv(glsl_source);
    return createComputeProgram(ctx, spirv, readonlyBuffers, readwriteBuffers);
}

void destroyComputeProgram(VulkanContext& ctx, ComputeProgram& prog) {
    ctx.device.destroyDescriptorPool(prog.descriptorPool);
    ctx.device.destroyPipeline(prog.pipeline);
    ctx.device.destroyPipelineLayout(prog.pipelineLayout);
    ctx.device.destroyDescriptorSetLayout(prog.descriptorLayout);
    ctx.device.destroyShaderModule(prog.shaderModule);
    prog.pipeline = nullptr;
    prog.pipelineLayout = nullptr;
    prog.descriptorLayout = nullptr;
    prog.descriptorPool = nullptr;
    prog.shaderModule = nullptr;
}

// dispatch compute commands
void runProgram(VulkanContext& ctx, ComputeProgram& prog, uint32_t n) {
    vk::CommandBufferAllocateInfo ai(ctx.commandPool, vk::CommandBufferLevel::ePrimary, 1);
    auto cmd = ctx.device.allocateCommandBuffers(ai)[0];

    cmd.begin(vk::CommandBufferBeginInfo{});
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, prog.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, prog.pipelineLayout, 0, prog.descriptorSet, {});
    uint32_t gs = 64, groups = (n + gs - 1) / gs;
    cmd.dispatch(groups, 1, 1);
    cmd.end();

    vk::Fence fence = ctx.device.createFence({});
    ctx.computeQueue.submit(vk::SubmitInfo(0, nullptr, 0, 1, &cmd), fence);

    [[maybe_unused]] auto rWait = ctx.device.waitForFences(fence, VK_TRUE, UINT64_MAX);
    // if you compile with VULKAN_HPP_NO_EXCEPTIONS=1, you can check:
    // if (rWait != vk::Result::eSuccess) throw std::runtime_error("waitForFences failed");

    ctx.device.destroyFence(fence);
    ctx.device.freeCommandBuffers(ctx.commandPool, cmd);
}