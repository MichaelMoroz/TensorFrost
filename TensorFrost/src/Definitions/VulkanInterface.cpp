#include "VulkanInterface.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <imgui.h>

#include "Backend/Vulkan.h"
#include "Backend/Window.h"

namespace py = pybind11;

namespace TensorFrost {
namespace {

bool isCContiguous(const py::buffer_info& info) {
    py::ssize_t stride = info.itemsize;
    for (py::ssize_t d = info.ndim - 1; d >= 0; --d) {
        if (info.strides[d] != stride) return false;
        stride *= info.shape[d];
    }
    return true;
}

}  // namespace

PyBuffer::PyBuffer(size_t count, size_t dtypeSize, bool readOnly)
        : ctx_(&getVulkanContext()),
            buffer_(createBuffer(count, dtypeSize, readOnly)),
            readOnly_(readOnly),
            dtypeSizeHint_(dtypeSize ? dtypeSize : 1),
            lastCount_(count),
            lastDtype_(py::none()) {}

PyBuffer::~PyBuffer() { release(); }

PyBuffer::PyBuffer(PyBuffer&& other) noexcept { moveFrom(std::move(other)); }

PyBuffer& PyBuffer::operator=(PyBuffer&& other) noexcept {
    if (this != &other) {
        release();
        moveFrom(std::move(other));
    }
    return *this;
}

bool PyBuffer::valid() const { return ctx_ && buffer_.buffer; }

size_t PyBuffer::byteSize() const { return buffer_.size; }

size_t PyBuffer::elementCapacity() const { return dtypeSizeHint_ ? buffer_.size / dtypeSizeHint_ : buffer_.size; }

bool PyBuffer::isReadOnly() const { return readOnly_; }

void PyBuffer::release() {
    if (ctx_ && buffer_.buffer) {
        destroyBuffer(buffer_);
    }
    buffer_ = {};
    ctx_ = nullptr;
    lastDtype_ = py::none();
    lastCount_ = 0;
    dtypeSizeHint_ = 0;
}

void PyBuffer::setData(const py::array& array, size_t offset) {
    ensureValid();
    auto info = array.request();
    if (!isCContiguous(info)) throw std::runtime_error("array must be C-contiguous");
    size_t nbytes = static_cast<size_t>(info.size) * static_cast<size_t>(info.itemsize);
    if (offset + nbytes > buffer_.size) throw std::out_of_range("write out of range");
    {
        py::gil_scoped_release release;
        setBufferData(buffer_, info.ptr, nbytes, offset);
    }
    lastDtype_ = array.dtype();
    lastCount_ = static_cast<size_t>(info.size);
    dtypeSizeHint_ = static_cast<size_t>(info.itemsize ? info.itemsize : 1);
}

py::array PyBuffer::getData(const py::object& dtypeArg, const py::object& countArg, size_t offset) const {
    ensureValid();
    if (offset > buffer_.size) throw std::out_of_range("offset out of range");

    py::dtype dtype = resolveDtype(dtypeArg);
    size_t itemsize = dtype.attr("itemsize").cast<size_t>();
    if (itemsize == 0) throw std::runtime_error("dtype itemsize cannot be zero");

    size_t available = buffer_.size - offset;
    size_t count = resolveCount(countArg, itemsize, available);
    size_t nbytes = count * itemsize;
    if (offset + nbytes > buffer_.size) throw std::out_of_range("read out of range");

    py::array out(dtype, py::array::ShapeContainer{ static_cast<py::ssize_t>(count) });
    auto info = out.request();
    {
        py::gil_scoped_release release;
        getBufferData(buffer_, info.ptr, nbytes, offset);
    }
    return out;
}

Buffer& PyBuffer::raw() {
    ensureValid();
    return buffer_;
}

const Buffer& PyBuffer::raw() const {
    ensureValid();
    return buffer_;
}

void PyBuffer::ensureValid() const {
    if (!valid()) throw std::runtime_error("Buffer has been released");
}

void PyBuffer::moveFrom(PyBuffer&& other) {
    ctx_ = other.ctx_;
    buffer_ = other.buffer_;
    readOnly_ = other.readOnly_;
    dtypeSizeHint_ = other.dtypeSizeHint_;
    lastCount_ = other.lastCount_;
    lastDtype_ = std::move(other.lastDtype_);
    other.ctx_ = nullptr;
    other.buffer_ = {};
    other.lastDtype_ = py::none();
}

py::dtype PyBuffer::resolveDtype(const py::object& dtypeArg) const {
    if (!dtypeArg.is_none()) {
        return py::reinterpret_borrow<py::dtype>(dtypeArg);
    }
    if (!lastDtype_.is_none()) {
        return py::reinterpret_borrow<py::dtype>(lastDtype_);
    }
    switch (dtypeSizeHint_) {
        case 2: return py::dtype::of<uint16_t>();
        case 4: return py::dtype::of<uint32_t>();
        case 8: return py::dtype::of<uint64_t>();
        default: return py::dtype::of<uint8_t>();
    }
}

size_t PyBuffer::resolveCount(const py::object& countArg, size_t itemsize, size_t available) const {
    if (!countArg.is_none()) {
        return countArg.cast<size_t>();
    }
    if (!lastDtype_.is_none() && itemsize == dtypeSizeHint_ && lastCount_ != 0) {
        return std::min(lastCount_, available / itemsize);
    }
    return available / itemsize;
}

PyComputeProgram::PyComputeProgram(ComputeProgram&& prog)
    : ctx_(&getVulkanContext()), program_(std::move(prog)) {}

PyComputeProgram::~PyComputeProgram() { release(); }

PyComputeProgram::PyComputeProgram(PyComputeProgram&& other) noexcept { moveFrom(std::move(other)); }

PyComputeProgram& PyComputeProgram::operator=(PyComputeProgram&& other) noexcept {
    if (this != &other) {
        release();
        moveFrom(std::move(other));
    }
    return *this;
}

void PyComputeProgram::run(const py::iterable& readonlyBuffers,
                           const py::iterable& readwriteBuffers,
                           uint32_t groupCount) {
    ensureValid();
    std::vector<Buffer*> ro;
    std::vector<Buffer*> rw;
    collectBuffers(readonlyBuffers, ro, "readonly");
    collectBuffers(readwriteBuffers, rw, "readwrite");
    if (ro.size() != program_.numRO || rw.size() != program_.numRW) {
        throw std::runtime_error("buffer count does not match program layout");
    }
    py::gil_scoped_release release;
    runProgram(program_, ro, rw, groupCount);
}

void PyComputeProgram::release() {
    if (ctx_ && program_.pipeline) {
        destroyComputeProgram(program_);
    }
    program_ = {};
    ctx_ = nullptr;
}

uint32_t PyComputeProgram::readonlyCount() const { return program_.numRO; }

uint32_t PyComputeProgram::readwriteCount() const { return program_.numRW; }

void PyComputeProgram::ensureValid() const {
    if (!ctx_ || !program_.pipeline) {
        throw std::runtime_error("ComputeProgram has been released");
    }
}

void PyComputeProgram::collectBuffers(const py::iterable& items,
                                      std::vector<Buffer*>& out,
                                      const char* label) {
    out.clear();
    for (auto obj : items) {
        try {
            py::handle handle(obj);
            auto* buf = handle.cast<PyBuffer*>();
            if (!buf) {
                throw py::cast_error("null buffer pointer");
            }
            out.push_back(&buf->raw());
        } catch (const py::cast_error&) {
            throw std::runtime_error(std::string("expected Buffer in ") + label + " list");
        }
    }
}

void PyComputeProgram::moveFrom(PyComputeProgram&& other) {
    ctx_ = other.ctx_;
    program_ = other.program_;
    other.ctx_ = nullptr;
    other.program_ = {};
}

PyWindow::PyWindow(int width, int height, const std::string& title)
    : ctx_(&getVulkanContext()), window_(createWindow(width, height, title.c_str())) {
    AttachWindowCallbacks(window_);
}

PyWindow::~PyWindow() = default;

PyWindow::PyWindow(PyWindow&& other) noexcept { moveFrom(std::move(other)); }

PyWindow& PyWindow::operator=(PyWindow&& other) noexcept {
    if (this != &other) {
        moveFrom(std::move(other));
    }
    return *this;
}

bool PyWindow::isOpen() const {
    ensureValid();
    return windowOpen(window_);
}

void PyWindow::drawBuffer(const PyBuffer& buffer, uint32_t width, uint32_t height, size_t offset) {
    ensureValid();
    py::gil_scoped_release release;
    ::drawBuffer(window_, buffer.raw(), width, height, offset);
}

void PyWindow::present() {
    ensureValid();
    py::gil_scoped_release release;
    ::drawBuffer(window_, vk::Buffer{}, window_.extent.width, window_.extent.height, 0);
}

py::tuple PyWindow::size() {
    ensureValid();
    int fbw = 0;
    int fbh = 0;
    glfwGetFramebufferSize(window_.wnd, &fbw, &fbh);
    if (fbw > 0 && fbh > 0) {
        window_.extent.width = static_cast<uint32_t>(fbw);
        window_.extent.height = static_cast<uint32_t>(fbh);
    }
    return py::make_tuple(window_.extent.width, window_.extent.height);
}

int PyWindow::format() const {
    ensureValid();
    return static_cast<int>(window_.format);
}

void PyWindow::close() {
    window_ = {};
    ctx_ = nullptr;
}

py::tuple PyWindow::imguiBegin(const std::string& name,
                               const std::optional<bool>& open,
                               int flags) {
    bindImGui();
    bool openValue = open.value_or(true);
    bool visible = ImGui::Begin(name.c_str(), open ? &openValue : nullptr, flags);
    return py::make_tuple(visible, open ? py::cast(openValue) : py::none());
}

void PyWindow::imguiEnd() {
    bindImGui();
    ImGui::End();
}

void PyWindow::imguiText(const std::string& text) {
    bindImGui();
    ImGui::TextUnformatted(text.c_str());
}

bool PyWindow::imguiButton(const std::string& label) {
    bindImGui();
    return ImGui::Button(label.c_str());
}

bool PyWindow::imguiCheckbox(const std::string& label, bool value) {
    bindImGui();
    bool v = value;
    ImGui::Checkbox(label.c_str(), &v);
    return v;
}

int PyWindow::imguiSliderInt(const std::string& label, int value, int min, int max) {
    bindImGui();
    int v = value;
    ImGui::SliderInt(label.c_str(), &v, min, max);
    return v;
}

float PyWindow::imguiSliderFloat(const std::string& label, float value, float min, float max) {
    bindImGui();
    float v = value;
    ImGui::SliderFloat(label.c_str(), &v, min, max);
    return v;
}

void PyWindow::imguiPlotLines(const std::string& label,
                              py::array_t<float, py::array::c_style | py::array::forcecast> values,
                              int valuesOffset,
                              const std::string& overlayText,
                              float scaleMin,
                              float scaleMax,
                              py::tuple graphSize,
                              int stride) {
    bindImGui();
    validateTupleSize(graphSize, 2, "graph_size");
    ImGui::PlotLines(
        label.c_str(),
        values.data(),
        static_cast<int>(values.size()),
        valuesOffset,
        overlayText.empty() ? nullptr : overlayText.c_str(),
        scaleMin,
        scaleMax,
        ImVec2(graphSize[0].cast<float>(), graphSize[1].cast<float>()),
        stride);
}

void PyWindow::imguiScaleAllSizes(float scale) {
    bindImGui();
    ImGui::GetStyle().ScaleAllSizes(scale);
}

void PyWindow::imguiAddBackgroundText(const std::string& text,
                                      py::tuple pos,
                                      py::tuple color) {
    bindImGui();
    validateTupleSize(pos, 2, "pos");
    validateTupleSize(color, 4, "color");
    ImGui::GetBackgroundDrawList()->AddText(
        ImVec2(pos[0].cast<float>(), pos[1].cast<float>()),
        ImColor(color[0].cast<float>(), color[1].cast<float>(), color[2].cast<float>(), color[3].cast<float>()),
        text.c_str());
}

void PyWindow::imguiSameLine(float offsetFromStartX, float spacing) {
    bindImGui();
    ImGui::SameLine(offsetFromStartX, spacing);
}

void PyWindow::imguiSeparator() {
    bindImGui();
    ImGui::Separator();
}

void PyWindow::imguiSpacing() {
    bindImGui();
    ImGui::Spacing();
}

void PyWindow::imguiIndent(float indentW) {
    bindImGui();
    ImGui::Indent(indentW);
}

void PyWindow::imguiUnindent(float indentW) {
    bindImGui();
    ImGui::Unindent(indentW);
}

bool PyWindow::imguiBeginChild(const std::string& id,
                               const py::object& size,
                               bool border,
                               int flags) {
    bindImGui();
    ImVec2 vecSize = objectToVec2(size, "size");
    return ImGui::BeginChild(id.c_str(), vecSize, border, flags);
}

void PyWindow::imguiEndChild() {
    bindImGui();
    ImGui::EndChild();
}

void PyWindow::imguiTextWrapped(const std::string& text) {
    bindImGui();
    ImGui::TextWrapped("%s", text.c_str());
}

void PyWindow::imguiTextColored(py::tuple color,
                                const std::string& text) {
    bindImGui();
    ImVec4 col = tupleToVec4(color, "color");
    ImGui::TextColored(col, "%s", text.c_str());
}

void PyWindow::imguiBulletText(const std::string& text) {
    bindImGui();
    ImGui::BulletText("%s", text.c_str());
}

std::tuple<bool, std::string> PyWindow::imguiInputText(const std::string& label,
                                                       const std::string& value,
                                                       size_t bufferLength,
                                                       int flags) {
    bindImGui();
    size_t minimum = value.size() + 1;
    size_t capacity = bufferLength ? std::max(bufferLength, minimum) : std::max(minimum, value.size() + 256);
    if (capacity == 0) capacity = 1;
    std::vector<char> buffer(capacity, '\0');
    std::copy(value.begin(), value.end(), buffer.begin());
    bool edited = ImGui::InputText(label.c_str(), buffer.data(), buffer.size(), flags);
    std::string result(buffer.data());
    return std::make_tuple(edited, std::move(result));
}

int PyWindow::imguiInputInt(const std::string& label, int value, int step, int stepFast, int flags) {
    bindImGui();
    int v = value;
    ImGui::InputInt(label.c_str(), &v, step, stepFast, flags);
    return v;
}

float PyWindow::imguiInputFloat(const std::string& label,
                                float value,
                                float step,
                                float stepFast,
                                const std::string& format,
                                int flags) {
    bindImGui();
    float v = value;
    ImGui::InputFloat(label.c_str(), &v, step, stepFast, format.c_str(), flags);
    return v;
}

std::tuple<bool, py::tuple> PyWindow::imguiColorEdit3(const std::string& label,
                                                      py::tuple color,
                                                      int flags) {
    bindImGui();
    validateTupleSize(color, 3, "color");
    float col[3] = {
        color[0].cast<float>(),
        color[1].cast<float>(),
        color[2].cast<float>()
    };
    bool changed = ImGui::ColorEdit3(label.c_str(), col, flags);
    return std::make_tuple(changed, py::make_tuple(col[0], col[1], col[2]));
}

std::tuple<bool, py::tuple> PyWindow::imguiColorEdit4(const std::string& label,
                                                      py::tuple color,
                                                      int flags) {
    bindImGui();
    validateTupleSize(color, 4, "color");
    float col[4] = {
        color[0].cast<float>(),
        color[1].cast<float>(),
        color[2].cast<float>(),
        color[3].cast<float>()
    };
    bool changed = ImGui::ColorEdit4(label.c_str(), col, flags);
    return std::make_tuple(changed, py::make_tuple(col[0], col[1], col[2], col[3]));
}

bool PyWindow::imguiBeginMainMenuBar() {
    bindImGui();
    return ImGui::BeginMainMenuBar();
}

void PyWindow::imguiEndMainMenuBar() {
    bindImGui();
    ImGui::EndMainMenuBar();
}

bool PyWindow::imguiBeginMenuBar() {
    bindImGui();
    return ImGui::BeginMenuBar();
}

void PyWindow::imguiEndMenuBar() {
    bindImGui();
    ImGui::EndMenuBar();
}

bool PyWindow::imguiBeginMenu(const std::string& label, bool enabled) {
    bindImGui();
    return ImGui::BeginMenu(label.c_str(), enabled);
}

void PyWindow::imguiEndMenu() {
    bindImGui();
    ImGui::EndMenu();
}

bool PyWindow::imguiMenuItem(const std::string& label,
                             const py::object& shortcut,
                             bool selected,
                             bool enabled) {
    bindImGui();
    std::string shortcutValue;
    const char* shortcutPtr = nullptr;
    if (!shortcut.is_none()) {
        shortcutValue = shortcut.cast<std::string>();
        shortcutPtr = shortcutValue.c_str();
    }
    return ImGui::MenuItem(label.c_str(), shortcutPtr, selected, enabled);
}

void PyWindow::imguiOpenPopup(const std::string& strId, int popupFlags) {
    bindImGui();
    ImGui::OpenPopup(strId.c_str(), popupFlags);
}

bool PyWindow::imguiBeginPopup(const std::string& strId, int flags) {
    bindImGui();
    return ImGui::BeginPopup(strId.c_str(), flags);
}

std::tuple<bool, py::object> PyWindow::imguiBeginPopupModal(const std::string& name,
                                                            const py::object& open,
                                                            int flags) {
    bindImGui();
    bool openValue = open.is_none() ? true : open.cast<bool>();
    bool visible = ImGui::BeginPopupModal(name.c_str(), open.is_none() ? nullptr : &openValue, flags);
    if (open.is_none()) {
        return std::make_tuple(visible, py::none());
    }
    return std::make_tuple(visible, py::cast(openValue));
}

void PyWindow::imguiEndPopup() {
    bindImGui();
    ImGui::EndPopup();
}

void PyWindow::imguiCloseCurrentPopup() {
    bindImGui();
    ImGui::CloseCurrentPopup();
}

void PyWindow::imguiPushStyleColor(int idx, py::tuple color) {
    bindImGui();
    ImVec4 col = tupleToVec4(color, "color");
    ImGui::PushStyleColor(idx, col);
}

void PyWindow::imguiPopStyleColor(int count) {
    bindImGui();
    ImGui::PopStyleColor(count);
}

void PyWindow::imguiPushStyleVarFloat(int idx, float value) {
    bindImGui();
    ImGui::PushStyleVar(idx, value);
}

void PyWindow::imguiPushStyleVarVec2(int idx, py::tuple value) {
    bindImGui();
    ImVec2 vec = objectToVec2(value, "value");
    ImGui::PushStyleVar(idx, vec);
}

void PyWindow::imguiPopStyleVar(int count) {
    bindImGui();
    ImGui::PopStyleVar(count);
}

float PyWindow::imguiGetFontGlobalScale() {
    bindImGui();
    return ImGui::GetIO().FontGlobalScale;
}

void PyWindow::imguiSetFontGlobalScale(float scale) {
    bindImGui();
    ImGui::GetIO().FontGlobalScale = scale;
}

py::tuple PyWindow::imguiGetStyleColorVec4(int idx) {
    bindImGui();
    ImVec4 col = ImGui::GetStyle().Colors[idx];
    return vec4ToTuple(col);
}

void PyWindow::imguiSetStyleColorVec4(int idx, py::tuple color) {
    bindImGui();
    ImVec4 col = tupleToVec4(color, "color");
    ImGui::GetStyle().Colors[idx] = col;
}

py::tuple PyWindow::mousePosition() {
    ensureValid();
    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos(window_.wnd, &x, &y);
    return py::make_tuple(x, y);
}

bool PyWindow::isMouseButtonPressed(int button) {
    ensureValid();
    return glfwGetMouseButton(window_.wnd, button) == GLFW_PRESS;
}

bool PyWindow::imguiWantCaptureMouse() const {
    ensureValid();
    if (!window_.imguiContext) {
        return false;
    }
    ImGui::SetCurrentContext(window_.imguiContext);
    return ImGui::GetIO().WantCaptureMouse;
}

py::tuple PyWindow::consumeScrollDelta() {
    ensureValid();
    double dx = window_.scrollDeltaX;
    double dy = window_.scrollDeltaY;
    window_.scrollDeltaX = 0.0;
    window_.scrollDeltaY = 0.0;
    return py::make_tuple(dx, dy);
}

void PyWindow::ensureValid() const {
    if (!window_.wnd) {
        throw std::runtime_error("Window has been closed");
    }
}

void PyWindow::moveFrom(PyWindow&& other) {
    ctx_ = other.ctx_;
    window_ = std::move(other.window_);
    other.ctx_ = nullptr;
    if (window_.wnd) {
        TFWindowDetail::RegisterScrollContext(window_.wnd, &window_);
    }
}

ImGuiContext* PyWindow::bindImGui() {
    ensureValid();
    EnsureImGuiFrame(window_);
    if (!window_.imguiContext) {
        throw std::runtime_error("ImGui context is not initialized for this window");
    }
    ImGui::SetCurrentContext(window_.imguiContext);
    return window_.imguiContext;
}

void PyWindow::validateTupleSize(const py::tuple& tpl, size_t expected, const char* name) {
    if (tpl.size() != expected) {
        throw std::invalid_argument(std::string("Expected tuple of size ") + std::to_string(expected) + " for " + name);
    }
}

ImVec2 PyWindow::objectToVec2(const py::object& obj, const char* name) {
    if (obj.is_none()) {
        return ImVec2(0.0f, 0.0f);
    }
    py::tuple tpl = obj.cast<py::tuple>();
    validateTupleSize(tpl, 2, name);
    return ImVec2(tpl[0].cast<float>(), tpl[1].cast<float>());
}

ImVec4 PyWindow::tupleToVec4(const py::tuple& tpl, const char* name) {
    validateTupleSize(tpl, 4, name);
    return ImVec4(tpl[0].cast<float>(),
                  tpl[1].cast<float>(),
                  tpl[2].cast<float>(),
                  tpl[3].cast<float>());
}

py::tuple PyWindow::vec4ToTuple(const ImVec4& vec) {
    return py::make_tuple(vec.x, vec.y, vec.z, vec.w);
}

PyComputeProgram MakeComputeProgramFromGLSL(const std::string& source,
                                            uint32_t roCount,
                                            uint32_t rwCount) {
    return PyComputeProgram(createComputeProgramFromGLSL(source, roCount, rwCount));
}

PyComputeProgram MakeComputeProgramFromSlang(const std::string& moduleName,
                                             const std::string& source,
                                             const std::string& entry,
                                             uint32_t roCount,
                                             uint32_t rwCount) {
    return PyComputeProgram(createComputeProgramFromSlang(moduleName, source, entry, roCount, rwCount));
}

}  // namespace TensorFrost
