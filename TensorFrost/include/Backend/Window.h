#pragma once
#include "Vulkan.h"
#include <vector>
#include <algorithm>
#include <stdexcept>

struct WindowContext {
    GLFWwindow* wnd{};
    vk::Instance instance;
    vk::PhysicalDevice phys;
    vk::Device device;
    uint32_t presentFam{};
    vk::Queue queue;

    vk::SurfaceKHR surface;
    vk::SwapchainKHR swapchain;
    std::vector<vk::Image> images;
    vk::Format format{};
    vk::Extent2D extent{};
    vk::CommandPool pool;
    vk::CommandBuffer cmd;
    vk::Semaphore semImage, semDone;
    vk::Fence fence;

    WindowContext() = default;
    WindowContext(const WindowContext&) = delete;
    WindowContext& operator=(const WindowContext&) = delete;

    WindowContext(WindowContext&& o) noexcept { moveFrom(std::move(o)); }
    WindowContext& operator=(WindowContext&& o) noexcept {
        if (this != &o) { cleanup(); moveFrom(std::move(o)); }
        return *this;
    }

    ~WindowContext() { cleanup(); }

private:
    void moveFrom(WindowContext&& o) {
        wnd=o.wnd; o.wnd=nullptr;
        instance=o.instance; o.instance=nullptr;
        phys=o.phys; o.phys=nullptr;
        device=o.device; o.device=nullptr;
        presentFam=o.presentFam; o.presentFam=0;
        queue=o.queue; o.queue=nullptr;
        surface=o.surface; o.surface=nullptr;
        swapchain=o.swapchain; o.swapchain=nullptr;
        images=std::move(o.images);
        format=o.format; o.format=vk::Format{};
        extent=o.extent; o.extent=vk::Extent2D{};
        pool=o.pool; o.pool=nullptr;
        cmd=o.cmd; o.cmd=nullptr;
        semImage=o.semImage; o.semImage=nullptr;
        semDone=o.semDone; o.semDone=nullptr;
        fence=o.fence; o.fence=nullptr;
    }

    void cleanup() {
        if (!wnd && !device) return;     // already moved/clean
        // don’t terminate GLFW here; only destroy this window
        if (device) {
            (void)device.waitIdle();
            if (fence)     device.destroyFence(fence),   fence=nullptr;
            if (semDone)   device.destroySemaphore(semDone), semDone=nullptr;
            if (semImage)  device.destroySemaphore(semImage), semImage=nullptr;
            if (cmd)       device.freeCommandBuffers(pool, cmd), cmd=nullptr;
            if (pool)      device.destroyCommandPool(pool), pool=nullptr;
            if (swapchain) device.destroySwapchainKHR(swapchain), swapchain=nullptr;
        }
        if (surface)  instance.destroySurfaceKHR(surface), surface=nullptr;
        if (wnd) { glfwDestroyWindow(wnd); wnd=nullptr; }
        // leave GLFW alive; app can call glfwTerminate() once at shutdown if it wants
        device=nullptr; instance=nullptr; queue=nullptr;
    }
};

WindowContext createWindow(VulkanContext& vctx, int width, int height, const char* title);
bool windowOpen(const WindowContext& ctx);
void drawBuffer(WindowContext& ctx, vk::Buffer src, uint32_t width, uint32_t height, vk::DeviceSize offset = 0);
void drawBuffer(WindowContext& ctx, const Buffer& b, uint32_t w, uint32_t h, size_t offset = 0);
