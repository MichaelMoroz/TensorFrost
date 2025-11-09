#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <GLFW/glfw3.h>

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
    uint32_t numRO = 0, numRW = 0;
    uint32_t pushConstantSize = 0;
};

struct ComputeBindings {
    vk::DescriptorSet set{};
};

// --- cache key + hash (as before)
struct DSKey {
    vk::DescriptorSetLayout layout{};
    std::vector<VkBuffer> bufs;
    std::vector<VkDeviceSize> sizes;
    bool operator==(const DSKey& o) const {
        return layout==o.layout && bufs==o.bufs && sizes==o.sizes;
    }
};
struct DSKeyHash {
    size_t operator()(DSKey const& k) const noexcept {
        auto mix = [](size_t& s, uint64_t v){ s ^= std::hash<uint64_t>{}(v) + 0x9e3779b97f4a7c15ULL + (s<<6) + (s>>2); };
        size_t seed = 0;
        mix(seed, (uint64_t)VkDescriptorSetLayout(k.layout));
        for (auto b : k.bufs)  mix(seed, (uint64_t)b);
        for (auto sz: k.sizes) mix(seed, (uint64_t)sz);
        return seed;
    }
};

// --- cached entry
struct CachedDS {
    vk::DescriptorSet set{};
    std::vector<VkBuffer> buffers;  // for invalidation
    uint64_t lastUseTick = 0;
    uint64_t useCount = 0;
};


struct CachedBuf {
    Buffer buf;
    uint64_t lastUse=0;
    uint64_t useCount=0;
};

// Holds instance, physical device, logical device, queue and command pool.
struct VulkanContext {
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    uint32_t queueFamilyIndex = UINT32_MAX;      // compute
    uint32_t graphicsFamilyIndex = UINT32_MAX;   // present candidate
    vk::Device device;
    vk::Queue computeQueue;
    vk::Queue presentQueue;
    vk::CommandPool commandPool;

    vk::DescriptorPool descriptorPool;
    std::unordered_map<DSKey, CachedDS, DSKeyHash> dsCache;
    size_t   dsCacheCapacity = 256; // must be ≤ pool maxSets
    uint64_t dsUseTick = 1;

    std::unordered_map<size_t, std::vector<CachedBuf>> bufferCache; // key: exact byte size
    size_t   bufferCacheCapacity = 128; // total entries across all sizes
    uint64_t bufferUseTick = 1;

    VulkanContext();
    ~VulkanContext();
};

Buffer createBuffer(size_t count, size_t dtypeSize, bool readOnly);
void destroyBuffer(Buffer& buf);
void setBufferData(Buffer& buf, const void* src, size_t bytes, size_t offset = 0);
void getBufferData(const Buffer& buf, void* dst, size_t bytes, size_t offset = 0);

ComputeProgram createComputeProgramFromSlang(const std::string& moduleName,
    const std::string& source, const std::string& entry,
    uint32_t roCount, uint32_t rwCount, uint32_t pushConstantSize = 0);
void destroyComputeProgram(ComputeProgram& prog);

void runProgram(const ComputeProgram& prog,
                const std::vector<Buffer*>& readonlyBuffers,
                const std::vector<Buffer*>& readwriteBuffers,
                uint32_t groupCount,
                const void* pushConstants,
                size_t pushConstantSize);

VulkanContext& getVulkanContext();
