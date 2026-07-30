# D013 implementation report

D013 implements two direct CPU modules. `buckets.py` derives the locked 17-shape family from the strict configuration, scales it proportionally, and returns either a complete cover-resize assignment or a scan-friendly rejection reason. `image_ops.py` applies Pillow EXIF transpose and RGB conversion, executes the assignment with Lanczos resize, and uses a caller-provided seed for the crop offsets.

The implementation performs no fallback routing: if no target fits without upscale, or the selected nearest-aspect cover crop retains less than the configured threshold, the sample is rejected. It does not access dataset payloads outside synthetic tests or add service/registry abstractions.
