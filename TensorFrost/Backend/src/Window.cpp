#include "Backend/Vulkan.h"
#include "Backend/Window.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <array>
#include <memory>
#include <mutex>
#include <string>

namespace {
std::unique_ptr<WindowContext> gWindow;
std::mutex gWindowMutex;

void CheckVkResult(VkResult err) {
    if (err == VK_SUCCESS) return;
    throw std::runtime_error("ImGui Vulkan backend error: " + std::to_string(err));
}

void EnsureFramebuffers(WindowContext& ctx) {
    if (ctx.framebuffers.size() == ctx.images.size() && !ctx.framebuffers.empty()) return;

    if (!ctx.renderPass) {
        vk::AttachmentDescription colorAttachment{};
        colorAttachment.format = ctx.format;
        colorAttachment.samples = vk::SampleCountFlagBits::e1;
        colorAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        colorAttachment.initialLayout = vk::ImageLayout::eColorAttachmentOptimal;
        colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

        vk::AttachmentReference colorAttachmentRef{0, vk::ImageLayout::eColorAttachmentOptimal};

        vk::SubpassDescription subpass{};
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        vk::SubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependency.srcAccessMask = {};
        dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

        vk::RenderPassCreateInfo rpci{};
        rpci.attachmentCount = 1;
        rpci.pAttachments = &colorAttachment;
        rpci.subpassCount = 1;
        rpci.pSubpasses = &subpass;
        rpci.dependencyCount = 1;
        rpci.pDependencies = &dependency;

        ctx.renderPass = ctx.device.createRenderPass(rpci);
    }

    if (ctx.imageViews.size() != ctx.images.size()) {
        for (auto view : ctx.imageViews) ctx.device.destroyImageView(view);
        ctx.imageViews.clear();
        ctx.imageViews.reserve(ctx.images.size());
        for (auto image : ctx.images) {
            vk::ImageViewCreateInfo viewInfo{};
            viewInfo.image = image;
            viewInfo.viewType = vk::ImageViewType::e2D;
            viewInfo.format = ctx.format;
            viewInfo.components = vk::ComponentMapping();
            viewInfo.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            ctx.imageViews.push_back(ctx.device.createImageView(viewInfo));
        }
    }

    for (auto fb : ctx.framebuffers) ctx.device.destroyFramebuffer(fb);
    ctx.framebuffers.clear();
    ctx.framebuffers.reserve(ctx.imageViews.size());
    for (auto view : ctx.imageViews) {
        vk::FramebufferCreateInfo fbci{};
        fbci.renderPass = ctx.renderPass;
        fbci.attachmentCount = 1;
        fbci.pAttachments = &view;
        fbci.width = ctx.extent.width;
        fbci.height = ctx.extent.height;
        fbci.layers = 1;
        ctx.framebuffers.push_back(ctx.device.createFramebuffer(fbci));
    }
}

void EnsureImGui(WindowContext& ctx, uint32_t imageCount) {
    if (ctx.imguiContext) return;

    EnsureFramebuffers(ctx);

    std::array<vk::DescriptorPoolSize, 11> poolSizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformTexelBuffer, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageTexelBuffer, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBufferDynamic, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBufferDynamic, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eInputAttachment, 1000}
    };

    vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                          1000 * static_cast<uint32_t>(poolSizes.size()),
                                          static_cast<uint32_t>(poolSizes.size()),
                                          poolSizes.data());
    ctx.imguiPool = ctx.device.createDescriptorPool(poolInfo);

    IMGUI_CHECKVERSION();
    ctx.imguiContext = ImGui::CreateContext();
    ImGui::SetCurrentContext(ctx.imguiContext);
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(ctx.wnd, true);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = ctx.instance;
    initInfo.PhysicalDevice = ctx.phys;
    initInfo.Device = ctx.device;
    initInfo.QueueFamily = ctx.presentFam;
    initInfo.Queue = ctx.queue;
    initInfo.Subpass = 0;
    initInfo.MinImageCount = imageCount;
    initInfo.ImageCount = imageCount;
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.Allocator = nullptr;
    initInfo.PipelineCache = VK_NULL_HANDLE;
    initInfo.DescriptorPool = ctx.imguiPool;
    initInfo.CheckVkResultFn = CheckVkResult;
    initInfo.RenderPass = static_cast<VkRenderPass>(ctx.renderPass);

    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        throw std::runtime_error("ImGui_ImplVulkan_Init failed");
    }

    if (!ImGui_ImplVulkan_CreateFontsTexture()) {
        throw std::runtime_error("ImGui_ImplVulkan_CreateFontsTexture failed");
    }
}

void StartImGuiFrame(WindowContext& ctx) {
    if (!ctx.imguiContext || ctx.imguiFrameActive) return;
    ImGui::SetCurrentContext(ctx.imguiContext);
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ctx.imguiFrameActive = true;
}
} // namespace

void ReleaseImGui(WindowContext& ctx) {
    if (!ctx.imguiContext) return;
    ImGui::SetCurrentContext(ctx.imguiContext);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(ctx.imguiContext);
    ctx.imguiContext = nullptr;
    ctx.imguiFrameActive = false;
}

WindowContext createWindow(int width, int height, const char* title) {
    auto& vctx = getVulkanContext();
    if (!glfwInit()) throw std::runtime_error("glfwInit");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    WindowContext ctx{};
    ctx.wnd     = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!ctx.wnd) throw std::runtime_error("glfwCreateWindow");

    ctx.instance = vctx.instance;
    ctx.phys     = vctx.physicalDevice;
    ctx.device   = vctx.device;

    // create surface on the shared instance
    VkSurfaceKHR raw{};
    if (glfwCreateWindowSurface(ctx.instance, ctx.wnd, nullptr, &raw) != VK_SUCCESS)
        throw std::runtime_error("glfw surface");
    ctx.surface = raw;

    // pick a present-capable family; prefer the graphics family we provisioned
    if (ctx.phys.getSurfaceSupportKHR(vctx.graphicsFamilyIndex, ctx.surface)) {
        ctx.presentFam = vctx.graphicsFamilyIndex;
        ctx.queue      = vctx.presentQueue;
    } else if (ctx.phys.getSurfaceSupportKHR(vctx.queueFamilyIndex, ctx.surface)) {
        ctx.presentFam = vctx.queueFamilyIndex;
        ctx.queue      = vctx.computeQueue; // uncommon but legal if it supports present
    } else {
        throw std::runtime_error("device queues do not support presenting to this surface");
    }

    auto caps = ctx.phys.getSurfaceCapabilitiesKHR(ctx.surface);
    auto fmts = ctx.phys.getSurfaceFormatsKHR(ctx.surface);
    vk::SurfaceFormatKHR sf = fmts.empty()
        ? vk::SurfaceFormatKHR{vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear}
        : fmts[0];
    if (!(caps.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferDst))
        throw std::runtime_error("swapchain missing TRANSFER_DST");

    int fbw, fbh; glfwGetFramebufferSize(ctx.wnd, &fbw, &fbh);
    ctx.extent = vk::Extent2D{
        uint32_t(std::clamp(fbw, int(caps.minImageExtent.width),  int(caps.maxImageExtent.width))),
        uint32_t(std::clamp(fbh, int(caps.minImageExtent.height), int(caps.maxImageExtent.height)))
    };
    ctx.format = sf.format;

    uint32_t imageCount = std::max(caps.minImageCount, 2u);
    if (caps.maxImageCount) imageCount = std::min(imageCount, caps.maxImageCount);

    ctx.swapchain = ctx.device.createSwapchainKHR({
        {}, ctx.surface, imageCount, sf.format, sf.colorSpace, ctx.extent, 1,
        vk::ImageUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive, 0, nullptr,
        caps.currentTransform, vk::CompositeAlphaFlagBitsKHR::eOpaque,
        vk::PresentModeKHR::eFifo, VK_TRUE, {}
    });
    ctx.images = ctx.device.getSwapchainImagesKHR(ctx.swapchain);

    ctx.pool = ctx.device.createCommandPool({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, ctx.presentFam});
    ctx.cmd  = ctx.device.allocateCommandBuffers({ctx.pool, vk::CommandBufferLevel::ePrimary, 1})[0];
    ctx.semImage = ctx.device.createSemaphore({});
    ctx.semDone  = ctx.device.createSemaphore({});
    ctx.fence    = ctx.device.createFence({});

    EnsureImGui(ctx, imageCount);
    StartImGuiFrame(ctx);

    return ctx;
}

bool windowOpen(const WindowContext &ctx) {
    return ctx.wnd && !glfwWindowShouldClose(ctx.wnd);
}

void drawBuffer(WindowContext &ctx, vk::Buffer src, uint32_t width, uint32_t height, vk::DeviceSize offset) {
    glfwPollEvents();

    auto acq = ctx.device.acquireNextImageKHR(ctx.swapchain, UINT64_MAX, ctx.semImage, {});
    if (acq.result == vk::Result::eSuboptimalKHR) {} // continue
    if (acq.result == vk::Result::eErrorOutOfDateKHR) return; // ignore; recreate swapchain if you want
    uint32_t idx = acq.value;

    ctx.cmd.reset({});
    ctx.cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

    vk::ImageMemoryBarrier toTransfer({}, vk::AccessFlagBits::eTransferWrite,
                                      vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                                      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                      ctx.images[idx], range);
    ctx.cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eTransfer,
                            {}, nullptr, nullptr, toTransfer);

    if (src) {
        vk::BufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        copy.imageExtent = vk::Extent3D{ width, height, 1 };
        ctx.cmd.copyBufferToImage(src, ctx.images[idx], vk::ImageLayout::eTransferDstOptimal, 1, &copy);
    }

    vk::ImageMemoryBarrier toColor(src ? vk::AccessFlagBits::eTransferWrite : vk::AccessFlagBits{},
                                   vk::AccessFlagBits::eColorAttachmentWrite,
                                   src ? vk::ImageLayout::eTransferDstOptimal : vk::ImageLayout::eUndefined,
                                   vk::ImageLayout::eColorAttachmentOptimal,
                                   VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                   ctx.images[idx], range);
    ctx.cmd.pipelineBarrier(src ? vk::PipelineStageFlagBits::eTransfer : vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            {}, nullptr, nullptr, toColor);

    EnsureFramebuffers(ctx);

    if (!src) {
        vk::ClearColorValue clearColor(std::array<float,4>{0.f, 0.f, 0.f, 1.f});
        std::array<vk::ImageSubresourceRange, 1> ranges{range};
        ctx.cmd.clearColorImage(ctx.images[idx], vk::ImageLayout::eColorAttachmentOptimal, clearColor, ranges);
    }

    vk::RenderPassBeginInfo rpBegin{};
    rpBegin.renderPass = ctx.renderPass;
    rpBegin.framebuffer = ctx.framebuffers[idx];
    rpBegin.renderArea.offset = vk::Offset2D{0, 0};
    rpBegin.renderArea.extent = ctx.extent;
    vk::ClearValue clearValue{};
    rpBegin.clearValueCount = 1;
    rpBegin.pClearValues = &clearValue;

    ctx.cmd.beginRenderPass(rpBegin, vk::SubpassContents::eInline);

    if (ctx.imguiContext && ctx.imguiFrameActive) {
        ImGui::SetCurrentContext(ctx.imguiContext);
        ImGui::Render();
        ImDrawData* drawData = ImGui::GetDrawData();
        if (drawData && drawData->CmdListsCount > 0) {
            ImGui_ImplVulkan_RenderDrawData(drawData, static_cast<VkCommandBuffer>(ctx.cmd));
        }
        ctx.imguiFrameActive = false;
    }

    ctx.cmd.endRenderPass();

    vk::ImageMemoryBarrier toPresent(vk::AccessFlagBits::eColorAttachmentWrite, {},
                                     vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
                                     VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                     ctx.images[idx], range);
    ctx.cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits::eBottomOfPipe,
                            {}, nullptr, nullptr, toPresent);

    ctx.cmd.end();

    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

    (void)ctx.device.resetFences(ctx.fence);
    ctx.queue.submit({vk::SubmitInfo(1, &ctx.semImage, &waitStage, 1, &ctx.cmd, 1, &ctx.semDone)}, ctx.fence);
    (void)ctx.device.waitForFences(ctx.fence, VK_TRUE, UINT64_MAX);

    try {
        (void)ctx.queue.presentKHR({1, &ctx.semDone, 1, &ctx.swapchain, &idx});
    } catch (const vk::OutOfDateKHRError&) {
        // ignore
    }

    StartImGuiFrame(ctx);
}

void drawBuffer(WindowContext &ctx, const Buffer &b, uint32_t w, uint32_t h, size_t offset) {
    // optional sanity check:
    if (offset + size_t(w)*size_t(h)*4 > b.size) throw std::out_of_range("buffer too small");
    drawBuffer(ctx, b.buffer, w, h, offset);
}

WindowContext* GetWindow() {
    std::scoped_lock lock(gWindowMutex);
    return gWindow.get();
}

WindowContext& RequireWindow() {
    auto* wnd = GetWindow();
    if (!wnd) throw std::runtime_error("Window not created. Call ShowWindow() first.");
    return *wnd;
}

ImGuiContext* GetImGuiContext() {
    auto* wnd = GetWindow();
    return wnd ? wnd->imguiContext : nullptr;
}

void EnsureImGuiFrame(WindowContext& ctx) {
    StartImGuiFrame(ctx);
}

void ShowWindow(int width, int height, const char* title) {
    std::scoped_lock lock(gWindowMutex);
    if (gWindow && windowOpen(*gWindow)) return;
    gWindow = std::make_unique<WindowContext>(createWindow(width, height, title));
}

void HideWindow() {
    std::scoped_lock lock(gWindowMutex);
    gWindow.reset();
}

void RenderFrame(const Buffer* buffer, uint32_t width, uint32_t height, size_t offset) {
    auto& ctx = RequireWindow();
    EnsureImGuiFrame(ctx);

    uint32_t w = width ? width : ctx.extent.width;
    uint32_t h = height ? height : ctx.extent.height;

    if (buffer) {
        drawBuffer(ctx, buffer->buffer, w, h, offset);
    } else {
        drawBuffer(ctx, vk::Buffer{}, w, h, offset);
    }
}

void RenderFrame(const Buffer* buffer) {
    auto& ctx = RequireWindow();
    RenderFrame(buffer, ctx.extent.width, ctx.extent.height, 0);
}

bool WindowShouldClose() {
    std::scoped_lock lock(gWindowMutex);
    if (!gWindow) return true;
    return !windowOpen(*gWindow);
}

std::pair<double, double> GetMousePosition() {
    std::scoped_lock lock(gWindowMutex);
    if (!gWindow || !gWindow->wnd) return {0.0, 0.0};
    double x{}, y{};
    glfwGetCursorPos(gWindow->wnd, &x, &y);
    return {x, y};
}

std::pair<int, int> GetWindowSize() {
    std::scoped_lock lock(gWindowMutex);
    if (!gWindow || !gWindow->wnd) return {0, 0};
    int w{}, h{};
    glfwGetWindowSize(gWindow->wnd, &w, &h);
    return {w, h};
}

bool IsMouseButtonPressed(int button) {
    std::scoped_lock lock(gWindowMutex);
    if (!gWindow || !gWindow->wnd) return false;
    return glfwGetMouseButton(gWindow->wnd, button) == GLFW_PRESS;
}

bool IsKeyPressed(int key) {
    std::scoped_lock lock(gWindowMutex);
    if (!gWindow || !gWindow->wnd) return false;
    return glfwGetKey(gWindow->wnd, key) == GLFW_PRESS;
}
