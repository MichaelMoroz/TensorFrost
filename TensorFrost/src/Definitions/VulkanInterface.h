#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#include "Backend/Vulkan.h"
#include "Backend/Window.h"

namespace TensorFrost {

class PyBuffer {
public:
    PyBuffer(size_t count, size_t dtypeSize, bool readOnly);
    ~PyBuffer();

    PyBuffer(const PyBuffer&) = delete;
    PyBuffer& operator=(const PyBuffer&) = delete;

    PyBuffer(PyBuffer&& other) noexcept;
    PyBuffer& operator=(PyBuffer&& other) noexcept;

    bool valid() const;
    size_t byteSize() const;
    size_t elementCapacity() const;
    bool isReadOnly() const;

    void release();

    void setData(const pybind11::array& array, size_t offset);
    pybind11::array getData(const pybind11::object& dtypeArg,
                            const pybind11::object& countArg,
                            size_t offset) const;

    Buffer& raw();
    const Buffer& raw() const;

private:
    void ensureValid() const;
    void moveFrom(PyBuffer&& other);
    pybind11::dtype resolveDtype(const pybind11::object& dtypeArg) const;
    size_t resolveCount(const pybind11::object& countArg, size_t itemsize, size_t available) const;

    VulkanContext* ctx_{};
    Buffer buffer_{};
    bool readOnly_{};
    size_t dtypeSizeHint_{};
    size_t lastCount_{};
    pybind11::object lastDtype_;
};

class PyComputeProgram {
public:
    explicit PyComputeProgram(ComputeProgram&& prog);
    ~PyComputeProgram();

    PyComputeProgram(const PyComputeProgram&) = delete;
    PyComputeProgram& operator=(const PyComputeProgram&) = delete;

    PyComputeProgram(PyComputeProgram&& other) noexcept;
    PyComputeProgram& operator=(PyComputeProgram&& other) noexcept;

    void run(const pybind11::iterable& readonlyBuffers,
             const pybind11::iterable& readwriteBuffers,
             uint32_t numInvocations);

    void release();

    uint32_t readonlyCount() const;
    uint32_t readwriteCount() const;

private:
    void ensureValid() const;
    static void collectBuffers(const pybind11::iterable& items,
                               std::vector<Buffer*>& out,
                               const char* label);
    void moveFrom(PyComputeProgram&& other);

    VulkanContext* ctx_{};
    ComputeProgram program_{};
};

class PyWindow {
public:
    PyWindow(int width, int height, const std::string& title);
    ~PyWindow();

    PyWindow(const PyWindow&) = delete;
    PyWindow& operator=(const PyWindow&) = delete;

    PyWindow(PyWindow&& other) noexcept;
    PyWindow& operator=(PyWindow&& other) noexcept;

    bool isOpen() const;
    void drawBuffer(const PyBuffer& buffer, uint32_t width, uint32_t height, size_t offset);
    void present();
    pybind11::tuple size();
    int format() const;
    void close();

    pybind11::tuple imguiBegin(const std::string& name,
                               const std::optional<bool>& open,
                               int flags);
    void imguiEnd();
    void imguiText(const std::string& text);
    bool imguiButton(const std::string& label);
    bool imguiCheckbox(const std::string& label, bool value);
    int imguiSliderInt(const std::string& label, int value, int min, int max);
    float imguiSliderFloat(const std::string& label, float value, float min, float max);
    void imguiPlotLines(const std::string& label,
                        pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> values,
                        int valuesOffset,
                        const std::string& overlayText,
                        float scaleMin,
                        float scaleMax,
                        pybind11::tuple graphSize,
                        int stride);
    void imguiScaleAllSizes(float scale);
    void imguiAddBackgroundText(const std::string& text,
                                pybind11::tuple pos,
                                pybind11::tuple color);

private:
    void ensureValid() const;
    void moveFrom(PyWindow&& other);
    ImGuiContext* bindImGui();
    static void validateTupleSize(const pybind11::tuple& tpl, size_t expected, const char* name);

    VulkanContext* ctx_{};
    WindowContext window_{};
};

PyComputeProgram MakeComputeProgramFromGLSL(const std::string& source,
                                            uint32_t roCount,
                                            uint32_t rwCount);

PyComputeProgram MakeComputeProgramFromSlang(const std::string& moduleName,
                                             const std::string& source,
                                             const std::string& entry,
                                             uint32_t roCount,
                                             uint32_t rwCount);

}  // namespace TensorFrost
