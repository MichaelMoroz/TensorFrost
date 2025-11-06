#include "VulkanInterface.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

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
                           uint32_t numInvocations) {
    ensureValid();
    std::vector<Buffer*> ro;
    std::vector<Buffer*> rw;
    collectBuffers(readonlyBuffers, ro, "readonly");
    collectBuffers(readwriteBuffers, rw, "readwrite");
    if (ro.size() != program_.numRO || rw.size() != program_.numRW) {
        throw std::runtime_error("buffer count does not match program layout");
    }
    py::gil_scoped_release release;
    runProgram(program_, ro, rw, numInvocations);
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
    : ctx_(&getVulkanContext()), window_(createWindow(width, height, title.c_str())) {}

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

py::tuple PyWindow::size() const {
    ensureValid();
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

void PyWindow::ensureValid() const {
    if (!window_.wnd) {
        throw std::runtime_error("Window has been closed");
    }
}

void PyWindow::moveFrom(PyWindow&& other) {
    ctx_ = other.ctx_;
    window_ = std::move(other.window_);
    other.ctx_ = nullptr;
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
