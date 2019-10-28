g++ -O3 -std=gnu++11 -o mnist_on_CPU_fix16_cpp mnist_on_CPU_fix16.cpp ./../layers_cpp/array_printf_fix16.cpp ./../layers_c/conv2d.c ./../layers_c/max_pooling2d.c ./../layers_c/padding2d.c ./../layers_c/up_sampling2d.c
g++ -O3 -std=gnu++11 -o mnist_on_CPU_fp32_cpp mnist_on_CPU_fp32.cpp ./../layers_cpp/array_printf_float32.cpp ./../layers_c/conv2d.c ./../layers_c/max_pooling2d.c ./../layers_c/padding2d.c ./../layers_c/up_sampling2d.c
./mnist_on_CPU_fix16_cpp
./mnist_on_CPU_fp32_cpp