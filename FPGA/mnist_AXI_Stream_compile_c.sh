gcc -std=gnu11 -o mnist_fix16_AXI_Stream_c mnist_fix16_AXI_Stream.c ./../layers_c/array_printf_fix16.c ./../layers_c/conv2d.c ./../layers_c/max_pooling2d.c ./../layers_c/padding2d.c ./../layers_c/up_sampling2d.c
gcc -std=gnu11 -o mnist_float32_AXI_Stream_c mnist_float32_AXI_Stream.c ./../layers_c/array_printf_float32.c ./../layers_c/conv2d.c ./../layers_c/max_pooling2d.c ./../layers_c/padding2d.c ./../layers_c/up_sampling2d.c
./mnist_fix16_AXI_Stream_c
./mnist_float32_AXI_Stream_c