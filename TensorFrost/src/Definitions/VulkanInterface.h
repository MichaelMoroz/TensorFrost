#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#include "Backend/Vulkan.h"
#include "Backend/Window.h"

struct ImVec2;
struct ImVec4;

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
    void imguiSameLine(float offsetFromStartX, float spacing);
    void imguiSeparator();
    void imguiSpacing();
    void imguiIndent(float indentW);
    void imguiUnindent(float indentW);
    bool imguiBeginChild(const std::string& id,
                         const pybind11::object& size,
                         bool border,
                         int flags);
    void imguiEndChild();
    void imguiTextWrapped(const std::string& text);
    void imguiTextColored(pybind11::tuple color,
                          const std::string& text);
    void imguiBulletText(const std::string& text);
    std::tuple<bool, std::string> imguiInputText(const std::string& label,
                                                 const std::string& value,
                                                 size_t bufferLength,
                                                 int flags);
    int imguiInputInt(const std::string& label, int value, int step, int stepFast, int flags);
    float imguiInputFloat(const std::string& label,
                          float value,
                          float step,
                          float stepFast,
                          const std::string& format,
                          int flags);
    std::tuple<bool, pybind11::tuple> imguiColorEdit3(const std::string& label,
                                                      pybind11::tuple color,
                                                      int flags);
    std::tuple<bool, pybind11::tuple> imguiColorEdit4(const std::string& label,
                                                      pybind11::tuple color,
                                                      int flags);
    bool imguiBeginMainMenuBar();
    void imguiEndMainMenuBar();
    bool imguiBeginMenuBar();
    void imguiEndMenuBar();
    bool imguiBeginMenu(const std::string& label, bool enabled);
    void imguiEndMenu();
    bool imguiMenuItem(const std::string& label,
                       const pybind11::object& shortcut,
                       bool selected,
                       bool enabled);
    void imguiOpenPopup(const std::string& strId, int popupFlags);
    bool imguiBeginPopup(const std::string& strId, int flags);
    std::tuple<bool, pybind11::object> imguiBeginPopupModal(const std::string& name,
                                                            const pybind11::object& open,
                                                            int flags);
    void imguiEndPopup();
    void imguiCloseCurrentPopup();
    void imguiPushStyleColor(int idx, pybind11::tuple color);
    void imguiPopStyleColor(int count);
    void imguiPushStyleVarFloat(int idx, float value);
    void imguiPushStyleVarVec2(int idx, pybind11::tuple value);
    void imguiPopStyleVar(int count);
    float imguiGetFontGlobalScale();
    void imguiSetFontGlobalScale(float scale);
    pybind11::tuple imguiGetStyleColorVec4(int idx);
    void imguiSetStyleColorVec4(int idx, pybind11::tuple color);

    pybind11::tuple mousePosition();
    bool isMouseButtonPressed(int button);
    bool imguiWantCaptureMouse() const;
    pybind11::tuple consumeScrollDelta();

private:
    void ensureValid() const;
    void moveFrom(PyWindow&& other);
    ImGuiContext* bindImGui();
    static void validateTupleSize(const pybind11::tuple& tpl, size_t expected, const char* name);
    static ImVec2 objectToVec2(const pybind11::object& obj, const char* name);
    static ImVec4 tupleToVec4(const pybind11::tuple& tpl, const char* name);
    static pybind11::tuple vec4ToTuple(const ImVec4& vec);

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
