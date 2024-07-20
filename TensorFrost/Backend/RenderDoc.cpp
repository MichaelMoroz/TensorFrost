#include "RenderDoc.h"
#include <iostream>
#include <renderdoc_app.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#include <sys/wait.h>
#endif

namespace TensorFrost {
    RENDERDOC_API_1_4_2* RDCAPI = nullptr;

    void LoadRDCAPI()
    {
        if (RDCAPI)
            return;

#if defined(_WIN32)
        if (HMODULE mod = GetModuleHandleA("renderdoc.dll"))
        {
            std::cout << "renderdoc.dll successfully loaded" << std::endl;
            pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
            RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_4_2, (void**)&RDCAPI);
        }
#else
        if (void* mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD))
        {
            std::cout << "librenderdoc.so successfully loaded" << std::endl;
            pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
            RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_4_2, (void**)&RDCAPI);
        }
#endif
    }

    void StartRenderDocCapture()
    {
        LoadRDCAPI();

        if (RDCAPI)
        {
            std::cout << "RenderDoc capture started" << std::endl;
            RDCAPI->StartFrameCapture(NULL, NULL);
        }
    }

    void EndRenderDocCapture()
    {
        if (RDCAPI)
        {
            std::cout << "RenderDoc capture ended" << std::endl;
            RDCAPI->EndFrameCapture(NULL, NULL);
        }
    }
}  // namespace TensorFrost