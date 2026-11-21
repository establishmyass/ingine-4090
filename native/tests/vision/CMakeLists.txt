# Image input: CPU vision kernels against FP64 oracles, preprocessing and the vision tower against the NumPy
# reference (tools/vision_reference.py goldens in tests/golden/vision).
ingine_add_test(NAME vision_kernels SOURCES test_vision.cpp LIBS ingine_vision ARGS kernels ORACLE)
ingine_add_test(NAME vision_tower SOURCES test_vision.cpp LIBS ingine_vision ARGS tower MODEL TIMEOUT 600)
