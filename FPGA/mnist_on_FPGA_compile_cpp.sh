# g++ -std=gnu++11 -o mnist_fix16_AXI_Stream_cpp mnist_fix16_AXI_Stream.cpp ./../layers_cpp/array_printf_fix16.cpp ./../layers_cpp/conv2d.cpp ./../layers_cpp/max_pooling2d.cpp ./../layers_cpp/padding2d.cpp ./../layers_cpp/up_sampling2d.cpp
g++ -std=gnu++11 -o mnist_fix16_AXI_Stream_cpp mnist_fix16_AXI_Stream.cpp ./../layers_cpp/array_printf_fix16.cpp ./../layers_c/conv2d.c ./../layers_c/max_pooling2d.c ./../layers_c/padding2d.c ./../layers_c/up_sampling2d.c
# g++ -std=gnu++11 -o mnist_float32_AXI_Stream_cpp mnist_float32_AXI_Stream.cpp ./../layers_cpp/array_printf_float32.cpp ./../layers_cpp/conv2d.cpp ./../layers_cpp/max_pooling2d.cpp ./../layers_cpp/padding2d.cpp ./../layers_cpp/up_sampling2d.cpp
g++ -std=gnu++11 -o mnist_float32_AXI_Stream_cpp mnist_float32_AXI_Stream.cpp ./../layers_cpp/array_printf_float32.cpp ./../layers_c/conv2d.c ./../layers_c/max_pooling2d.c ./../layers_c/padding2d.c ./../layers_c/up_sampling2d.c
./mnist_fix16_AXI_Stream_cpp
./mnist_float32_AXI_Stream_cpp