#include "Backend/RenderDoc.h"

#include <renderdoc_app.h>

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <string>

#include "Backend/Vulkan.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace TensorFrost {
namespace {
RENDERDOC_API_1_4_2* gRenderDocApi = nullptr;

void LoadRenderDoc()
{
    static bool loggedUnavailable = false;

    if (gRenderDocApi) {
        return;
    }

#if defined(_WIN32)
    const char* moduleNames[] = {"renderdoc.dll", "renderdoccmd.dll"};

    for (const char* moduleName : moduleNames) {
        HMODULE mod = GetModuleHandleA(moduleName);
        if (!mod) {
            continue;
        }

        auto getApi = reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(mod, "RENDERDOC_GetAPI"));
        if (getApi &&
            getApi(eRENDERDOC_API_Version_1_4_2, reinterpret_cast<void**>(&gRenderDocApi)) == 1) {
            std::cout << "RenderDoc API loaded from " << moduleName << std::endl;
            loggedUnavailable = false;
            break;
        }
    }
#else
    if (void* mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD)) {
        auto getApi = reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(mod, "RENDERDOC_GetAPI"));
        if (getApi && getApi(eRENDERDOC_API_Version_1_4_2, reinterpret_cast<void**>(&gRenderDocApi)) == 1) {
            std::cout << "RenderDoc API loaded" << std::endl;
            loggedUnavailable = false;
            dlclose(mod);
            return;
        }
        dlclose(mod);
    }
#endif

    if (!gRenderDocApi && !loggedUnavailable) {
        std::cout << "RenderDoc API not available" << std::endl;
        loggedUnavailable = true;
    }
}
}  // namespace

bool IsRenderDocAvailable()
{
    LoadRenderDoc();
    return gRenderDocApi != nullptr;
}

void StartRenderDocCapture()
{
    if (!IsRenderDocAvailable()) {
        std::cout << "RenderDoc not available; start capture skipped" << std::endl;
        return;
    }

    RENDERDOC_DevicePointer deviceHandle = nullptr;
    try {
        auto& ctx = getVulkanContext();
        VkInstance vkInstance = static_cast<VkInstance>(ctx.instance);
        deviceHandle = RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(vkInstance);
    } catch (const std::exception& e) {
        std::cout << "RenderDoc capture start warning: failed to get Vulkan instance (" << e.what() << ")" << std::endl;
    }

    gRenderDocApi->StartFrameCapture(deviceHandle, nullptr);
    if (gRenderDocApi->IsFrameCapturing && gRenderDocApi->IsFrameCapturing() == 1) {
        std::cout << "RenderDoc capture start requested" << std::endl;
    } else {
        std::cout << "RenderDoc capture start did not begin (IsFrameCapturing=0)" << std::endl;
    }
}

std::string EndRenderDocCapture(bool launchReplayUI)
{
    if (!IsRenderDocAvailable()) {
        std::cout << "RenderDoc not available; end capture skipped" << std::endl;
        return {};
    }

    RENDERDOC_DevicePointer deviceHandle = nullptr;
    try {
        auto& ctx = getVulkanContext();
        VkInstance vkInstance = static_cast<VkInstance>(ctx.instance);
        deviceHandle = RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(vkInstance);
    } catch (const std::exception& e) {
        std::cout << "RenderDoc capture end warning: failed to get Vulkan instance (" << e.what() << ")" << std::endl;
    }

    std::string capturePath;
    const uint32_t result = gRenderDocApi->EndFrameCapture(deviceHandle, nullptr);
    if (result == 1) {
        std::cout << "RenderDoc capture end requested" << std::endl;
        if (gRenderDocApi->GetNumCaptures && gRenderDocApi->GetCapture) {
            const uint32_t numCaptures = gRenderDocApi->GetNumCaptures();
            if (numCaptures > 0) {
                const uint32_t captureIndex = numCaptures - 1;
                uint32_t pathLength = 0;
                if (gRenderDocApi->GetCapture(captureIndex, nullptr, &pathLength, nullptr) == 1 && pathLength > 0) {
                    capturePath.resize(pathLength);
                    if (gRenderDocApi->GetCapture(captureIndex, capturePath.data(), &pathLength, nullptr) == 1) {
                        if (!capturePath.empty() && capturePath.back() == '\0') {
                            capturePath.pop_back();
                        }
                        std::cout << "RenderDoc capture stored at: " << capturePath << std::endl;
                        if (launchReplayUI && !capturePath.empty()) {
                            if (gRenderDocApi->LaunchReplayUI) {
                                const std::string commandLine = "\"" + capturePath + "\"";
                                const uint32_t pid = gRenderDocApi->LaunchReplayUI(0, commandLine.c_str());
                                if (pid == 0) {
                                    std::cout << "RenderDoc replay UI launch failed" << std::endl;
                                } else {
                                    std::cout << "RenderDoc replay UI launched (PID " << pid << ")" << std::endl;
                                }
                            } else {
                                std::cout << "RenderDoc replay UI launch unavailable (API function missing)" << std::endl;
                            }
                        }
                    } else {
                        capturePath.clear();
                    }
                }
            }
        }
    } else {
        std::cout << "RenderDoc capture end failed (" << result << ")" << std::endl;
    }

    return capturePath;
}

}  // namespace TensorFrost
