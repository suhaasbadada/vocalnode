# Project Architecture
* This is a Voice AI platform combining microWakeWord and ChatterboxTTS.
* The core backend is built with Python and FastAPI. 
* Use `pip` for dependency management.

# Claude Code Agent Directives
* DO NOT execute `apt-get`, `brew install`, or modify system audio drivers directly. Output required system dependencies to `system_reqs.txt`.
* DO NOT read raw `.wav`, `.pt`, `.tflite`, or `.onnx` files into the context window.
* Write rigorous Pytest suites for all logic before implementing the main application code.