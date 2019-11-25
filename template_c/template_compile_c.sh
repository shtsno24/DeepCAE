sed -i 's/\r//' *.sh
gcc -O0 -o template_fix16 template_fix16.c ./../layers_c/array_printf_fix16.c ./../layers_c/conv2d.c ./../layers_c/max_pooling2d.c ./../layers_c/padding2d.c ./../layers_c/up_sampling2d.c ./../layers_c/depthwise_conv2d.c ./../layers_c/pointwise_conv2d.c ./../layers_c/separable_conv2d.c 
gcc -O0 -o template_float32 template_float32.c ./../layers_c/array_printf_float32.c ./../layers_c/conv2d.c ./../layers_c/max_pooling2d.c ./../layers_c/padding2d.c ./../layers_c/up_sampling2d.c ./../layers_c/depthwise_conv2d.c ./../layers_c/pointwise_conv2d.c ./../layers_c/separable_conv2d.c
./template_fix16
./template_float32