import unittest
from pathlib import Path

import numpy as np

import TensorFrost as tf


_SIMPLE_SLANG_SHADER = """[[vk::binding(0,0)]] StructuredBuffer<uint> InputBuffer : register(t0, space0);
[[vk::binding(1,0)]] RWStructuredBuffer<uint> OutputBuffer : register(u1, space0);

[numthreads(64, 1, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
        return;

    uint value = InputBuffer[0];
    OutputBuffer[0] = value + 1u;
}
"""

_REQUIRED_ARTIFACTS = (
    "slang.dll",
    "slang-glslang.dll",
    "spirv-opt.exe",
)


def _runtime_dir() -> Path:
    return Path(tf.__file__).resolve().parent


def _missing_runtime_artifacts() -> list[str]:
    runtime_dir = _runtime_dir()
    missing: list[str] = []
    for artifact in _REQUIRED_ARTIFACTS:
        release_candidate = runtime_dir / artifact
        debug_candidate = runtime_dir / release_candidate.with_name(release_candidate.stem + "d" + release_candidate.suffix).name
        if not release_candidate.exists() and not debug_candidate.exists():
            missing.append(artifact)
    return missing

def _should_skip_for_backend(exc: Exception) -> bool:
    message = str(exc).lower()
    keywords = (
        "glfw",
        "vulkan",
        "no physical devices",
        "no suitable",
        "device",
        "surface",
        "swapchain",
    )
    return any(token in message for token in keywords)


class SlangCompilationTest(unittest.TestCase):
    def test_compile_and_execute_simple_shader(self) -> None:
        thread_count = 1
        local_size = 64
        group_count = max((thread_count + local_size - 1) // local_size, 1)

        missing_artifacts = _missing_runtime_artifacts()
        if missing_artifacts:  # pragma: no cover - environment not staged
            pretty = ", ".join(missing_artifacts)
            self.skipTest(
                "Slang runtime components missing: "
                f"{pretty}. Re-run setup_python_env.cmd or rebuild the TensorFrost target to stage runtimes."
            )

        try:
            readonly_buffer = tf.createBuffer(thread_count, 4, True)
        except RuntimeError as exc:  # pragma: no cover - Vulkan not available
            self.skipTest(f"Vulkan buffer creation failed: {exc}")

        try:
            readwrite_buffer = tf.createBuffer(thread_count, 4, False)
        except RuntimeError as exc:  # pragma: no cover - Vulkan not available
            self.skipTest(f"Vulkan buffer creation failed: {exc}")

        try:
            program = tf.createComputeProgramFromSlang(
                "tensorfrost_test_shader",
                _SIMPLE_SLANG_SHADER,
                "csMain",
                ro_count=1,
                rw_count=1,
            )
        except RuntimeError as exc:
            if _should_skip_for_backend(exc):  # pragma: no cover - Vulkan not available
                self.skipTest(f"Slang compilation backend unavailable: {exc}")
            raise

        readonly_buffer.setData(np.array([7], dtype=np.uint32))
        readwrite_buffer.setData(np.zeros(1, dtype=np.uint32))

        program.run([readonly_buffer], [readwrite_buffer], group_count)

        result = readwrite_buffer.getData(np.dtype(np.uint32), thread_count)
        self.assertEqual(result.shape, (thread_count,))
        self.assertEqual(int(result[0]), 8)

    readonly_buffer = None
    readwrite_buffer = None
    program = None

    import gc

    gc.collect()


if __name__ == "__main__":
    unittest.main()
