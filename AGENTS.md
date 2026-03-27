# Agent Guide

Follow these expectations whenever you work in this repository:

1. **Full rebuild & virtual environment** — Run `setup_python_env.cmd` from the repo root. It configures the Python virtual environment and performs a clean rebuild so you start from a consistent state.
2. **Partial rebuilds** — Use CMake for incremental builds. Invoke the appropriate CMake build command (for example, `cmake --build <build-directory> --target <target>`) to rebuild only what you need.
3. **C++ changes** — Any edits under `TensorFrost/` or other C++ sources require a rebuild before the changes take effect.
4. **Python script changes** - After edits of python script, you should run them to make sure they work correctly. No recompilation needed.
5. **API validation** — After modifying functionality, run the relevant tests in the `tests/` folder to confirm the Python API still behaves as expected.
6. **Scenario validation** — Run the sample programs in the `examples/` folder to make sure the updated stack handles more complex end-to-end flows.
