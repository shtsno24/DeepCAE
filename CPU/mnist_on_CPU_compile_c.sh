gcc -O3 -o mnist_on_CPU_fix16_c mnist_on_CPU_fix16.c ./../layers_c/array_printf_fix16.c ./../layers_c/conv2d.c ./../layers_c/max_pooling2d.c ./../layers_c/padding2d.c ./../layers_c/up_sampling2d.c
gcc -O3 -o mnist_on_CPU_fp32_c mnist_on_CPU_fp32.c ./../layers_c/array_printf_float32.c ./../layers_c/conv2d.c ./../layers_c/max_pooling2d.c ./../layers_c/padding2d.c ./../layers_c/up_sampling2d.c 
./mnist_on_CPU_fix16_c
./mnist_on_CPU_fp32_c