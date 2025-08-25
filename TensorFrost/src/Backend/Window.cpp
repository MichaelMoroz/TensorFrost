#include "Backend/Vulkan.h"
#include "Backend/Window.h"

WindowContext createWindow(VulkanContext& vctx, int width, int height, const char* title) {
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

    vk::ImageMemoryBarrier toDst({}, vk::AccessFlagBits::eTransferWrite,
                                 vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                                 VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                 ctx.images[idx], {vk::ImageAspectFlagBits::eColor, 0,1, 0,1});
    ctx.cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, toDst);

    vk::BufferImageCopy copy{};
    copy.bufferOffset = offset;
    copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copy.imageExtent = vk::Extent3D{ width, height, 1 };
    ctx.cmd.copyBufferToImage(src, ctx.images[idx], vk::ImageLayout::eTransferDstOptimal, 1, &copy);

    vk::ImageMemoryBarrier toPresent(vk::AccessFlagBits::eTransferWrite, {},
                                     vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR,
                                     VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                     ctx.images[idx], {vk::ImageAspectFlagBits::eColor, 0,1, 0,1});
    ctx.cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, toPresent);

    ctx.cmd.end();

    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eTransfer;

    (void)ctx.device.resetFences(ctx.fence);
    ctx.queue.submit({vk::SubmitInfo(1, &ctx.semImage, &waitStage, 1, &ctx.cmd, 1, &ctx.semDone)}, ctx.fence);
    (void)ctx.device.waitForFences(ctx.fence, VK_TRUE, UINT64_MAX);

    try {
        (void)ctx.queue.presentKHR({1, &ctx.semDone, 1, &ctx.swapchain, &idx});
    } catch (const vk::OutOfDateKHRError&) {
        // ignore
    }
}

void drawBuffer(WindowContext &ctx, const Buffer &b, uint32_t w, uint32_t h, size_t offset) {
    // optional sanity check:
    if (offset + size_t(w)*size_t(h)*4 > b.size) throw std::out_of_range("buffer too small");
    drawBuffer(ctx, b.buffer, w, h, offset);
}
