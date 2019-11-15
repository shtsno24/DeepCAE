sed -i 's/\r//' *.sh
gcc -O0 -o mnist_on_CPU_fix16_c_Sep mnist_on_CPU_fix16_Sep.c ./../layers_c/array_printf_fix16.c ./../layers_c/conv2d.c ./../layers_c/max_pooling2d.c ./../layers_c/padding2d.c ./../layers_c/up_sampling2d.c ./../layers_c/depthwise_conv2d.c ./../layers_c/pointwise_conv2d.c ./../layers_c/separable_conv2d.c 
gcc -O0 -o mnist_on_CPU_fp32_c_Sep mnist_on_CPU_fp32_Sep.c ./../layers_c/array_printf_float32.c ./../layers_c/conv2d.c ./../layers_c/max_pooling2d.c ./../layers_c/padding2d.c ./../layers_c/up_sampling2d.c ./../layers_c/depthwise_conv2d.c ./../layers_c/pointwise_conv2d.c ./../layers_c/separable_conv2d.c
./mnist_on_CPU_fix16_c_Sep
./mnist_on_CPU_fp32_c_Sep