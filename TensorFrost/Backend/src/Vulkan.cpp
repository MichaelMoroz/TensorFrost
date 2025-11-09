#include "Backend/Vulkan.h"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#include <slang/slang.h>
#include <slang/slang-com-ptr.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
VulkanContext& getOrCreateGlobalContext() {
    static VulkanContext ctx{};
    return ctx;
}
}

VulkanContext& getVulkanContext() {
    return getOrCreateGlobalContext();
}

VulkanContext::VulkanContext() {
    if (!glfwInit()) throw std::runtime_error("GLFW init failed");
    if (!glfwVulkanSupported())
        throw std::runtime_error("GLFW: Vulkan loader not found. Install a Vulkan-capable GPU driver or the Vulkan SDK.");

    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    uint32_t extCount = 0;
    const char** extNames = glfwGetRequiredInstanceExtensions(&extCount);
    if (!extNames || extCount == 0) throw std::runtime_error("GLFW: Vulkan not supported");

    vk::ApplicationInfo appInfo("TensorFrost", 1, nullptr, 0, VK_API_VERSION_1_2);
    vk::InstanceCreateInfo instCreate({}, &appInfo, 0, nullptr, extCount, extNames);
    instance = vk::createInstance(instCreate);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

    // pick device + queue families (compute + graphics)
    auto devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) throw std::runtime_error("No physical devices");

    for (auto& pd : devices) {
        auto q = pd.getQueueFamilyProperties();
        int compute = -1, graphics = -1;
        for (uint32_t i = 0; i < q.size(); ++i) {
            auto f = q[i].queueFlags;
            if (compute  < 0 && (f & vk::QueueFlagBits::eCompute))  compute  = int(i);
            if (graphics < 0 && (f & vk::QueueFlagBits::eGraphics)) graphics = int(i);
        }
        if (compute >= 0 && graphics >= 0) {
            physicalDevice = pd;
            queueFamilyIndex    = uint32_t(compute);
            graphicsFamilyIndex = uint32_t(graphics);
            break;
        }
    }
    if (!physicalDevice) throw std::runtime_error("No suitable device");

    // device with both queues + swapchain
    float prio = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queues;
    queues.emplace_back(vk::DeviceQueueCreateInfo({}, queueFamilyIndex, 1, &prio));
    if (graphicsFamilyIndex != queueFamilyIndex)
        queues.emplace_back(vk::DeviceQueueCreateInfo({}, graphicsFamilyIndex, 1, &prio));

    // enable VK_KHR_swapchain (required for window/swapchain)
    const char* devExts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    vk::DeviceCreateInfo devCreate({}, (uint32_t)queues.size(), queues.data(),
                                   0, nullptr, 1, devExts);
    device = physicalDevice.createDevice(devCreate);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);

    computeQueue = device.getQueue(queueFamilyIndex, 0);
    presentQueue = device.getQueue(graphicsFamilyIndex, 0);

    vk::CommandPoolCreateInfo poolInfo({}, queueFamilyIndex);
    commandPool = device.createCommandPool(poolInfo);

    vk::DescriptorPoolSize sz(vk::DescriptorType::eStorageBuffer, 1024);
    vk::DescriptorPoolCreateInfo dp(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 256, 1, &sz);
    descriptorPool = device.createDescriptorPool(dp);
    dsCacheCapacity = 256;
}

static void evictSome(VulkanContext& ctx, size_t n) {
    if (ctx.dsCache.empty() || n == 0) return;
    std::vector<std::unordered_map<DSKey, CachedDS, DSKeyHash>::iterator> items;
    items.reserve(ctx.dsCache.size());
    for (auto it = ctx.dsCache.begin(); it != ctx.dsCache.end(); ++it) items.push_back(it);
    std::stable_sort(items.begin(), items.end(),
        [](auto a, auto b){ return a->second.lastUseTick < b->second.lastUseTick; }); // LRU first
    n = std::min(n, items.size());
    for (size_t i = 0; i < n; ++i) {
        auto it = items[i];
        if (it->second.set) ctx.device.freeDescriptorSets(ctx.descriptorPool, 1, &it->second.set);
        ctx.dsCache.erase(it);
    }
}

static void evictToCapacity(VulkanContext& ctx) {
    if (ctx.dsCache.size() > ctx.dsCacheCapacity)
        evictSome(ctx, ctx.dsCache.size() - ctx.dsCacheCapacity);
}

static void invalidateDescriptorCacheForBuffer(VulkanContext& ctx, VkBuffer buf) {
    std::vector<decltype(ctx.dsCache.begin())> dead;
    for (auto it = ctx.dsCache.begin(); it != ctx.dsCache.end(); ++it)
        if (std::find(it->second.buffers.begin(), it->second.buffers.end(), buf) != it->second.buffers.end())
            dead.push_back(it);
    for (auto it : dead) {
        if (it->second.set) ctx.device.freeDescriptorSets(ctx.descriptorPool, 1, &it->second.set);
        ctx.dsCache.erase(it);
    }
}

static void invalidateDescriptorCacheForLayout(VulkanContext& ctx, vk::DescriptorSetLayout layout) {
    std::vector<decltype(ctx.dsCache.begin())> dead;
    for (auto it = ctx.dsCache.begin(); it != ctx.dsCache.end(); ++it)
        if (it->first.layout == layout) dead.push_back(it);
    for (auto it : dead) {
        if (it->second.set) ctx.device.freeDescriptorSets(ctx.descriptorPool, 1, &it->second.set);
        ctx.dsCache.erase(it);
    }
}

// optional knobs
void setDescriptorCacheCapacity(VulkanContext& ctx, size_t cap) {
    ctx.dsCacheCapacity = cap;
    evictToCapacity(ctx);
}
void clearDescriptorCache(VulkanContext& ctx) {
    evictSome(ctx, ctx.dsCache.size());
}

// --- cached descriptor set retrieval
static vk::DescriptorSet getOrCreateSet(VulkanContext& ctx, const ComputeProgram& prog,
                                        const std::vector<Buffer*>& ro, const std::vector<Buffer*>& rw) {
    if (ro.size()!=prog.numRO || rw.size()!=prog.numRW) throw std::runtime_error("buffer count != program layout");

    DSKey key; key.layout = prog.descriptorLayout;
    key.bufs.reserve(ro.size()+rw.size());
    key.sizes.reserve(ro.size()+rw.size());
    for (auto* b : ro) { key.bufs.push_back(b->buffer); key.sizes.push_back(b->size); }
    for (auto* b : rw) { key.bufs.push_back(b->buffer); key.sizes.push_back(b->size); }

    if (auto it = ctx.dsCache.find(key); it != ctx.dsCache.end()) {
        it->second.lastUseTick = ++ctx.dsUseTick;
        it->second.useCount++;
        return it->second.set;
    }

    evictToCapacity(ctx);

    vk::DescriptorSet set{};
    for (int attempt = 0; attempt < 3; ++attempt) {
        try {
            vk::DescriptorSetAllocateInfo ai(ctx.descriptorPool, 1, &prog.descriptorLayout);
            set = ctx.device.allocateDescriptorSets(ai)[0];
            break;
        } catch (const vk::SystemError& e) {
            auto r = static_cast<vk::Result>(e.code().value());
            if (r == vk::Result::eErrorOutOfPoolMemory || r == vk::Result::eErrorFragmentedPool) {
                evictSome(ctx, std::max<size_t>(1, ctx.dsCache.size()/4));
            } else {
                throw;
            }
        }
    }
    if (!set) throw std::runtime_error("Descriptor set allocation failed");

    std::vector<vk::DescriptorBufferInfo> infos; infos.reserve(key.bufs.size());
    for (auto* b : ro) infos.emplace_back(b->buffer, 0, b->size);
    for (auto* b : rw) infos.emplace_back(b->buffer, 0, b->size);

    std::vector<vk::WriteDescriptorSet> writes; writes.reserve(infos.size());
    for (uint32_t i=0;i<infos.size();++i)
        writes.emplace_back(set, i, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[i]);
    ctx.device.updateDescriptorSets(writes, {});

    CachedDS meta;
    meta.set = set;
    meta.buffers = key.bufs;
    meta.lastUseTick = ++ctx.dsUseTick;
    meta.useCount = 1;
    ctx.dsCache.emplace(std::move(key), std::move(meta));
    return set;
}

static size_t bufferCacheCount(const VulkanContext& ctx) {
    size_t n = 0;
    for (auto& kv : ctx.bufferCache) n += kv.second.size();
    return n;
}

static void evictBuffers(VulkanContext& ctx, size_t n) {
    if (n == 0) return;

    struct Node { size_t key; vk::Buffer buf; uint64_t last; };
    std::vector<Node> nodes; nodes.reserve(bufferCacheCount(ctx));

    for (auto& kv : ctx.bufferCache)
        for (auto& e : kv.second)
            nodes.push_back(Node{ kv.first, e.buf.buffer, e.lastUse });

    std::stable_sort(nodes.begin(), nodes.end(),
                     [](const Node& a, const Node& b){ return a.last < b.last; });

    if (n > nodes.size()) n = nodes.size();

    for (size_t i = 0; i < n; ++i) {
        auto itMap = ctx.bufferCache.find(nodes[i].key);
        if (itMap == ctx.bufferCache.end()) continue;

        auto& vec = itMap->second;

        auto it = std::find_if(vec.begin(), vec.end(),
            [&](const CachedBuf& e){
                return e.buf.buffer == nodes[i].buf;
            });
        if (it == vec.end()) continue; // already evicted by an earlier iteration

        Buffer b = it->buf;
        vec.erase(it);
        if (vec.empty()) ctx.bufferCache.erase(itMap);

        if (b.buffer) ctx.device.destroyBuffer(b.buffer);
        if (b.memory) ctx.device.freeMemory(b.memory);
    }
}

static void evictBuffersToCapacity(VulkanContext& ctx) {
    size_t cnt = bufferCacheCount(ctx);
    if (cnt > ctx.bufferCacheCapacity) evictBuffers(ctx, cnt - ctx.bufferCacheCapacity);
}

void clearBufferCache(VulkanContext& ctx) { evictBuffers(ctx, bufferCacheCount(ctx)); }

void setBufferCacheCapacity(VulkanContext& ctx, size_t cap) { ctx.bufferCacheCapacity = cap; evictBuffersToCapacity(ctx); }

static bool takeBufferFromCache(VulkanContext& ctx, size_t bytes, Buffer& out) {
    auto it = ctx.bufferCache.find(bytes);
    if (it == ctx.bufferCache.end() || it->second.empty()) return false;
    auto e = std::move(it->second.back());
    it->second.pop_back();
    if (it->second.empty()) ctx.bufferCache.erase(it);
    out = e.buf; // handles copied; cache keeps no reference
    return true;
}

// create a storage buffer
Buffer createBuffer(size_t count, size_t dtypeSize, bool readOnly) {
    auto& ctx = getVulkanContext();
    Buffer buf{};
    buf.size = count * dtypeSize;

    if (takeBufferFromCache(ctx, buf.size, buf)) return buf;

    vk::BufferCreateInfo bci({}, buf.size,
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc |
        vk::BufferUsageFlagBits::eTransferDst);
    buf.buffer = ctx.device.createBuffer(bci);
    auto memReq = ctx.device.getBufferMemoryRequirements(buf.buffer);

    auto memProps = ctx.physicalDevice.getMemoryProperties();
    uint32_t memTypeIndex = UINT32_MAX;
    // Prefer device-local memory for GPU performance
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReq.memoryTypeBits & (1u<<i)) &&
            (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal))
        { memTypeIndex = i; break; }
    }
    // Fallback: pick any compatible memory type
    if (memTypeIndex == UINT32_MAX) {
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if (memReq.memoryTypeBits & (1u<<i)) { memTypeIndex = i; break; }
        }
    }
    if (memTypeIndex == UINT32_MAX) throw std::runtime_error("No compatible memory type for buffer");
    vk::MemoryAllocateInfo allocInfo(memReq.size, memTypeIndex);
    buf.memory = ctx.device.allocateMemory(allocInfo);
    ctx.device.bindBufferMemory(buf.buffer, buf.memory, 0);
    return buf;
}

void destroyBuffer(Buffer& buf) {
    auto& ctx = getVulkanContext();
    if (!buf.buffer) return;
    // logical death → drop any dsCache entries that reference it
    invalidateDescriptorCacheForBuffer(ctx, buf.buffer); // keep if you have the DS cache; else remove this line

    CachedBuf e;
    e.buf = buf;
    e.lastUse = ++ctx.bufferUseTick;
    e.useCount = 1;
    ctx.bufferCache[buf.size].push_back(std::move(e));
    buf.buffer = nullptr;
    buf.memory = nullptr;
    buf.size   = 0;
    evictBuffersToCapacity(ctx);
}

void setBufferData(Buffer& buf, const void* src, size_t bytes, size_t offset) {
    auto& ctx = getVulkanContext();
    if (offset + bytes > buf.size) throw std::out_of_range("write out of range");
    if (bytes == 0) return;

    // Create a temporary host-visible staging buffer for upload
    vk::BufferCreateInfo bci({}, bytes, vk::BufferUsageFlagBits::eTransferSrc);
    vk::Buffer staging = ctx.device.createBuffer(bci);
    auto memReq = ctx.device.getBufferMemoryRequirements(staging);

    auto memProps = ctx.physicalDevice.getMemoryProperties();
    uint32_t memTypeIndex = UINT32_MAX;
    // Prefer HOST_VISIBLE | HOST_COHERENT for simple map without flush; fallback to HOST_VISIBLE
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        auto f = memProps.memoryTypes[i].propertyFlags;
        if ((memReq.memoryTypeBits & (1u<<i)) &&
            (f & vk::MemoryPropertyFlagBits::eHostVisible) &&
            (f & vk::MemoryPropertyFlagBits::eHostCoherent))
        { memTypeIndex = i; break; }
    }
    if (memTypeIndex == UINT32_MAX) {
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            auto f = memProps.memoryTypes[i].propertyFlags;
            if ((memReq.memoryTypeBits & (1u<<i)) && (f & vk::MemoryPropertyFlagBits::eHostVisible))
            { memTypeIndex = i; break; }
        }
    }
    if (memTypeIndex == UINT32_MAX) {
        ctx.device.destroyBuffer(staging);
        throw std::runtime_error("No host-visible memory type for staging upload");
    }
    vk::DeviceMemory stagingMem = ctx.device.allocateMemory(vk::MemoryAllocateInfo(memReq.size, memTypeIndex));
    ctx.device.bindBufferMemory(staging, stagingMem, 0);

    // Map and copy data into staging (with non-coherent alignment handling)
    auto atom = ctx.physicalDevice.getProperties().limits.nonCoherentAtomSize;
    vk::DeviceSize mapOff = 0;
    vk::DeviceSize mapEnd = ((bytes + atom - 1) / atom) * atom;
    vk::DeviceSize mapSz  = mapEnd - mapOff;
    void* p = ctx.device.mapMemory(stagingMem, mapOff, mapSz);
    std::memcpy(p, src, bytes);
    // Flush if not coherent
    if (!(memProps.memoryTypes[memTypeIndex].propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
        vk::MappedMemoryRange rng(stagingMem, mapOff, mapSz);
        ctx.device.flushMappedMemoryRanges(rng);
    }
    ctx.device.unmapMemory(stagingMem);

    // Record and submit copy from staging to device-local buffer
    vk::CommandBufferAllocateInfo ai(ctx.commandPool, vk::CommandBufferLevel::ePrimary, 1);
    auto cmd = ctx.device.allocateCommandBuffers(ai)[0];
    cmd.begin(vk::CommandBufferBeginInfo{});
    vk::BufferCopy copy(0, offset, bytes);
    cmd.copyBuffer(staging, buf.buffer, 1, &copy);
    cmd.end();

    vk::Fence fence = ctx.device.createFence({});
    ctx.computeQueue.submit(vk::SubmitInfo(0, nullptr, 0, 1, &cmd), fence);
    [[maybe_unused]] auto rWait = ctx.device.waitForFences(fence, VK_TRUE, UINT64_MAX);
    ctx.device.destroyFence(fence);
    ctx.device.freeCommandBuffers(ctx.commandPool, cmd);

    // Destroy staging resources
    ctx.device.destroyBuffer(staging);
    ctx.device.freeMemory(stagingMem);
}

void getBufferData(const Buffer& buf, void* dst, size_t bytes, size_t offset) {
    auto& ctx = getVulkanContext();
    if (offset + bytes > buf.size) throw std::out_of_range("read out of range");
    if (bytes == 0) return;

    // Create a temporary host-visible staging buffer for download
    vk::BufferCreateInfo bci({}, bytes, vk::BufferUsageFlagBits::eTransferDst);
    vk::Buffer staging = ctx.device.createBuffer(bci);
    auto memReq = ctx.device.getBufferMemoryRequirements(staging);

    auto memProps = ctx.physicalDevice.getMemoryProperties();
    uint32_t memTypeIndex = UINT32_MAX;
    // Prefer HOST_VISIBLE | HOST_COHERENT; fallback to HOST_VISIBLE
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        auto f = memProps.memoryTypes[i].propertyFlags;
        if ((memReq.memoryTypeBits & (1u<<i)) &&
            (f & vk::MemoryPropertyFlagBits::eHostVisible) &&
            (f & vk::MemoryPropertyFlagBits::eHostCoherent))
        { memTypeIndex = i; break; }
    }
    if (memTypeIndex == UINT32_MAX) {
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            auto f = memProps.memoryTypes[i].propertyFlags;
            if ((memReq.memoryTypeBits & (1u<<i)) && (f & vk::MemoryPropertyFlagBits::eHostVisible))
            { memTypeIndex = i; break; }
        }
    }
    if (memTypeIndex == UINT32_MAX) {
        ctx.device.destroyBuffer(staging);
        throw std::runtime_error("No host-visible memory type for staging download");
    }
    vk::DeviceMemory stagingMem = ctx.device.allocateMemory(vk::MemoryAllocateInfo(memReq.size, memTypeIndex));
    ctx.device.bindBufferMemory(staging, stagingMem, 0);

    // Record and submit copy from device-local buffer to staging
    vk::CommandBufferAllocateInfo ai(ctx.commandPool, vk::CommandBufferLevel::ePrimary, 1);
    auto cmd = ctx.device.allocateCommandBuffers(ai)[0];
    cmd.begin(vk::CommandBufferBeginInfo{});
    vk::BufferCopy copy(offset, 0, bytes);
    cmd.copyBuffer(buf.buffer, staging, 1, &copy);
    cmd.end();

    vk::Fence fence = ctx.device.createFence({});
    ctx.computeQueue.submit(vk::SubmitInfo(0, nullptr, 0, 1, &cmd), fence);
    [[maybe_unused]] auto rWait = ctx.device.waitForFences(fence, VK_TRUE, UINT64_MAX);
    ctx.device.destroyFence(fence);
    ctx.device.freeCommandBuffers(ctx.commandPool, cmd);

    // Map and read back from staging (with non-coherent alignment handling)
    auto atom = ctx.physicalDevice.getProperties().limits.nonCoherentAtomSize;
    vk::DeviceSize mapOff = 0;
    vk::DeviceSize mapEnd = ((bytes + atom - 1) / atom) * atom;
    vk::DeviceSize mapSz  = mapEnd - mapOff;
    void* p = ctx.device.mapMemory(stagingMem, mapOff, mapSz);
    if (!(memProps.memoryTypes[memTypeIndex].propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
        vk::MappedMemoryRange rng(stagingMem, mapOff, mapSz);
        ctx.device.invalidateMappedMemoryRanges(rng);
    }
    std::memcpy(dst, p, bytes);
    ctx.device.unmapMemory(stagingMem);

    // Destroy staging resources
    ctx.device.destroyBuffer(staging);
    ctx.device.freeMemory(stagingMem);
}

VulkanContext::~VulkanContext() {
    // free cached sets first
    for (auto it = dsCache.begin(); it != dsCache.end(); ++it) {
        if (it->second.set) device.freeDescriptorSets(descriptorPool, 1, &it->second.set);
    }
    dsCache.clear();
    clearBufferCache(*this);
    device.destroyDescriptorPool(descriptorPool);
    device.destroyCommandPool(commandPool);
    device.destroy();
    instance.destroy();
}

struct SlangCompileResult {
    std::vector<uint32_t> spirv;
    uint32_t pushConstantSize = 0;
};

SlangCompileResult compileSlangToSpirv(const char* moduleName,
                                       const char* source,
                                       const char* entry,
                                       const char* profile /* e.g., "spirv_1_5" */) {
    Slang::ComPtr<slang::IGlobalSession> global;
    createGlobalSession(global.writeRef());

    slang::TargetDesc tgt{};
    tgt.format = SLANG_SPIRV;
    tgt.profile = global->findProfile(profile);

    slang::SessionDesc sd{};
    sd.targets = &tgt; sd.targetCount = 1;
#if defined(_RELWITHDEBINFO)
    std::array<slang::CompilerOptionEntry, 2> optionEntries{};
    optionEntries[0].name = slang::CompilerOptionName::DebugInformation;
    optionEntries[0].value.kind = slang::CompilerOptionValueKind::Int;
    optionEntries[0].value.intValue0 = SLANG_DEBUG_INFO_LEVEL_STANDARD;
    optionEntries[1].name = slang::CompilerOptionName::Optimization;
    optionEntries[1].value.kind = slang::CompilerOptionValueKind::Int;
    optionEntries[1].value.intValue0 = SLANG_OPTIMIZATION_LEVEL_NONE;
    sd.compilerOptionEntries = optionEntries.data();
    sd.compilerOptionEntryCount = static_cast<uint32_t>(optionEntries.size());
#endif

    Slang::ComPtr<slang::ISession> session;
    global->createSession(sd, session.writeRef());

    Slang::ComPtr<slang::IBlob> diag;
    Slang::ComPtr<slang::IModule> mod;
    mod = session->loadModuleFromSourceString(moduleName, moduleName, source, diag.writeRef());
    if (diag && diag->getBufferSize()) std::fprintf(stderr, "%s\n", (const char*)diag->getBufferPointer());
    if (!mod) throw std::runtime_error("slang: module load failed");

    Slang::ComPtr<slang::IEntryPoint> ep;
    mod->findEntryPointByName(entry, ep.writeRef());
    if (!ep) throw std::runtime_error("slang: entry not found");

    slang::IComponentType* parts[] = { mod.get(), ep.get() };
    Slang::ComPtr<slang::IComponentType> composed, linked;

    {
        Slang::ComPtr<slang::IBlob> d;
        SlangResult r = session->createCompositeComponentType(parts, 2, composed.writeRef(), d.writeRef());
        if (d && d->getBufferSize()) std::fprintf(stderr, "%s\n", (const char*)d->getBufferPointer());
        if (SLANG_FAILED(r)) throw std::runtime_error("slang: compose failed");
    }
    {
        Slang::ComPtr<slang::IBlob> d;
        SlangResult r = composed->link(linked.writeRef(), d.writeRef());
        if (d && d->getBufferSize()) std::fprintf(stderr, "%s\n", (const char*)d->getBufferPointer());
        if (SLANG_FAILED(r)) throw std::runtime_error("slang: link failed");
    }

    Slang::ComPtr<slang::IBlob> spirv;
    {
        Slang::ComPtr<slang::IBlob> d;
        SlangResult r = linked->getEntryPointCode(0, 0, spirv.writeRef(), d.writeRef());
        if (d && d->getBufferSize()) std::fprintf(stderr, "%s\n", (const char*)d->getBufferPointer());
        if (SLANG_FAILED(r)) throw std::runtime_error("slang: getEntryPointCode failed");
    }

    uint32_t pushConstantSize = 0;
    {
        Slang::ComPtr<slang::IBlob> diag;
        slang::ProgramLayout* layout = linked->getLayout(0, diag.writeRef());
        if (diag && diag->getBufferSize()) std::fprintf(stderr, "%s\n", (const char*)diag->getBufferPointer());
        if (!layout) throw std::runtime_error("slang: failed to obtain program layout");

        if (auto* globalLayout = layout->getGlobalParamsTypeLayout()) {
            size_t size = globalLayout->getSize(slang::ParameterCategory::PushConstantBuffer);
            pushConstantSize = std::max(pushConstantSize, static_cast<uint32_t>(size));
        }
        for (SlangUInt i = 0; i < layout->getEntryPointCount(); ++i) {
            if (auto* entry = layout->getEntryPointByIndex(i)) {
                if (auto* typeLayout = entry->getTypeLayout()) {
                    size_t size = typeLayout->getSize(slang::ParameterCategory::PushConstantBuffer);
                    pushConstantSize = std::max(pushConstantSize, static_cast<uint32_t>(size));
                }
            }
        }
    }

    size_t n = spirv->getBufferSize();
    auto* p = static_cast<const uint8_t*>(spirv->getBufferPointer());
    std::vector<uint32_t> out((n + 3) / 4);
    std::memcpy(out.data(), p, n);
    return {std::move(out), pushConstantSize};
}

ComputeBindings createBindings(VulkanContext& ctx, const ComputeProgram& prog,
                               const std::vector<Buffer*>& readonlyBuffers,
                               const std::vector<Buffer*>& readwriteBuffers) {
    if (readonlyBuffers.size() != prog.numRO || readwriteBuffers.size() != prog.numRW)
        throw std::runtime_error("buffer count != program layout");

    vk::DescriptorSetAllocateInfo ai(ctx.descriptorPool, 1, &prog.descriptorLayout);
    ComputeBindings b{};
    b.set = ctx.device.allocateDescriptorSets(ai)[0];

    std::vector<vk::DescriptorBufferInfo> infos;
    infos.reserve(prog.numRO + prog.numRW);
    for (auto* x : readonlyBuffers)  infos.emplace_back(x->buffer, 0, x->size);
    for (auto* x : readwriteBuffers) infos.emplace_back(x->buffer, 0, x->size);

    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(infos.size());
    for (uint32_t i = 0; i < infos.size(); ++i)
        writes.emplace_back(b.set, i, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[i]);

    ctx.device.updateDescriptorSets(writes, {});
    return b;
}

static ComputeProgram createComputeProgram(const std::vector<uint32_t>& spirv,
    uint32_t roCount, uint32_t rwCount, uint32_t pushConstantSize) {
    auto& ctx = getVulkanContext();

    ComputeProgram prog;
    prog.numRO = roCount; prog.numRW = rwCount;
    prog.pushConstantSize = pushConstantSize;

    if (prog.pushConstantSize) {
        auto limits = ctx.physicalDevice.getProperties().limits;
        if (prog.pushConstantSize > limits.maxPushConstantsSize) {
            throw std::runtime_error("push constant block exceeds device limit");
        }
    }

    vk::ShaderModuleCreateInfo smci({}, spirv.size() * sizeof(uint32_t), spirv.data());
    prog.shaderModule = ctx.device.createShaderModule(smci);

    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(roCount + rwCount);
    for (uint32_t b = 0; b < roCount + rwCount; ++b)
        bindings.emplace_back(b, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);

    vk::DescriptorSetLayoutCreateInfo dsInfo({}, bindings.size(), bindings.data());
    prog.descriptorLayout = ctx.device.createDescriptorSetLayout(dsInfo);

    vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eCompute, 0, prog.pushConstantSize);
    auto pushPtr = prog.pushConstantSize ? &pushRange : nullptr;
    uint32_t pushCount = prog.pushConstantSize ? 1u : 0u;

    vk::PipelineLayoutCreateInfo plInfo({}, 1, &prog.descriptorLayout, pushCount, pushPtr);
    prog.pipelineLayout = ctx.device.createPipelineLayout(plInfo);

    vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, prog.shaderModule, "main");
    vk::ComputePipelineCreateInfo cpInfo({}, stageInfo, prog.pipelineLayout);
    prog.pipeline = ctx.device.createComputePipeline({}, cpInfo).value;

    return prog;
}

ComputeProgram createComputeProgramFromSlang(const std::string& moduleName,
    const std::string& source, const std::string& entry, uint32_t roCount, uint32_t rwCount) {
    auto result = compileSlangToSpirv(moduleName.c_str(), source.c_str(), entry.c_str(), "spirv_1_5");
    return createComputeProgram(result.spirv, roCount, rwCount, result.pushConstantSize);
}

void destroyComputeProgram(ComputeProgram& prog) {
    auto& ctx = getVulkanContext();
    invalidateDescriptorCacheForLayout(ctx, prog.descriptorLayout);
    ctx.device.destroyPipeline(prog.pipeline);
    ctx.device.destroyPipelineLayout(prog.pipelineLayout);
    ctx.device.destroyDescriptorSetLayout(prog.descriptorLayout);
    ctx.device.destroyShaderModule(prog.shaderModule);
    prog = {};
}

void runProgram(const ComputeProgram& prog,
                const std::vector<Buffer*>& readonlyBuffers,
                const std::vector<Buffer*>& readwriteBuffers,
                uint32_t groupCount,
                const void* pushConstants,
                size_t pushConstantSize) {
    auto& ctx = getVulkanContext();
    auto set = getOrCreateSet(ctx, prog, readonlyBuffers, readwriteBuffers);

    vk::CommandBufferAllocateInfo ai(ctx.commandPool, vk::CommandBufferLevel::ePrimary, 1);
    auto cmd = ctx.device.allocateCommandBuffers(ai)[0];

    cmd.begin(vk::CommandBufferBeginInfo{});
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, prog.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, prog.pipelineLayout, 0, set, {});

    if (prog.pushConstantSize) {
        if (!pushConstants) {
            throw std::runtime_error("push constant payload missing");
        }
        if (pushConstantSize != prog.pushConstantSize) {
            throw std::runtime_error("push constant payload size mismatch");
        }
        cmd.pushConstants(prog.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                          prog.pushConstantSize, pushConstants);
    } else if (pushConstantSize != 0) {
        throw std::runtime_error("push constant payload provided but pipeline has none");
    }

    cmd.dispatch(groupCount, 1, 1);
    cmd.end();

    vk::Fence fence = ctx.device.createFence({});
    ctx.computeQueue.submit(vk::SubmitInfo(0, nullptr, 0, 1, &cmd), fence);
    [[maybe_unused]] auto rWait = ctx.device.waitForFences(fence, VK_TRUE, UINT64_MAX);
    ctx.device.destroyFence(fence);
    ctx.device.freeCommandBuffers(ctx.commandPool, cmd);
}

