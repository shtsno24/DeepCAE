# DepthwiseConv2D

## Output Layer Data and array shapes

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    depthwise_conv2d (DepthwiseC (None, 14, 8, 6)          96        
    =================================================================
    Total params: 96
    Trainable params: 96
    Non-trainable params: 0
    _________________________________________________________________
    weights[0].shape :  (3, 5, 6, 1)
    weights[1].shape :  (6,)
    input_img.shape :  (6, 16, 12)
    input_img_keras.shape :  (1, 16, 12, 6)
    output_img_keras.shape (1, 14, 8, 6)
    output_img.shape :  (1, 6, 14, 8)
    
    layer_name :  DepthwiseConv2D
    first_input_shape :  [16, 12, 6]
    padding :  valid
    stride :  [1, 1]
    kernel_size :  [3, 5]
    depth_multiplier :  1

## Json Output

    first_input_shape : Same as Conv2D
    padding : Same as Conv2D
    stride : Same as Conv2D
    kernel_size : Same as Conv2D
    depth_multiplier : output_depth can be expressed as input_depth * depth_multiplier.  
    To make the netowork generate simple, it would be fine to limit depth_multiplier to 1.  

## Weight format

    weight_shape : k_height, k_width, k_in_channel is as same as Conv2D.  
    However k_out_channel is depend on not output_array_shape,  
    but depth_multiplier.  

## Bias format

    bias_shape : Same as Conv2D, it depends on output_depth.
