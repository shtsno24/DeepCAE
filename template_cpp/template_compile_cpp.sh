sed -i 's/\r//' *.sh
g++ -O3 -std=gnu++11 -o template_fix16 template_fix16.cpp ./../layers_cpp/array_printf_fix16.cpp ./../layers_cpp/conv2d.cpp ./../layers_cpp/max_pooling2d.cpp ./../layers_cpp/padding2d.cpp ./../layers_cpp/up_sampling2d.cpp ./../layers_cpp/depthwise_conv2d.cpp ./../layers_cpp/pointwise_conv2d.cpp ./../layers_cpp/separable_conv2d.cpp 
g++ -O3 -std=gnu++11 -o template_float32 template_float32.cpp ./../layers_cpp/array_printf_float32.cpp ./../layers_cpp/conv2d.cpp ./../layers_cpp/max_pooling2d.cpp ./../layers_cpp/padding2d.cpp ./../layers_cpp/up_sampling2d.cpp ./../layers_cpp/depthwise_conv2d.cpp ./../layers_cpp/pointwise_conv2d.cpp ./../layers_cpp/separable_conv2d.cpp
./template_fix16
./template_float32