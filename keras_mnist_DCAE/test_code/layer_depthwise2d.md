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


Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
depthwise_conv2d (DepthwiseC (None, 7, 7, 3)           30        
=================================================================
Total params: 30
Trainable params: 30
Non-trainable params: 0
_________________________________________________________________
weights[0].shape :  (3, 3, 3, 1)
weights[1].shape :  (3,)
input_img.shape :  (3, 7, 7)
input_img_keras.shape :  (1, 7, 7, 3)
[[[ 1.  2.  3.  4.  5.  6.  7.]
  [ 2.  4.  6.  8. 10. 12. 14.]
  [ 3.  6.  9. 12. 15. 18. 21.]
  [ 4.  8. 12. 16. 20. 24. 28.]
  [ 5. 10. 15. 20. 25. 30. 35.]
  [ 6. 12. 18. 24. 30. 36. 42.]
  [ 7. 14. 21. 28. 35. 42. 49.]]

 [[ 1.  2.  3.  4.  5.  6.  7.]
  [ 2.  4.  6.  8. 10. 12. 14.]
  [ 3.  6.  9. 12. 15. 18. 21.]
  [ 4.  8. 12. 16. 20. 24. 28.]
  [ 5. 10. 15. 20. 25. 30. 35.]
  [ 6. 12. 18. 24. 30. 36. 42.]
  [ 7. 14. 21. 28. 35. 42. 49.]]

 [[ 1.  2.  3.  4.  5.  6.  7.]
  [ 2.  4.  6.  8. 10. 12. 14.]
  [ 3.  6.  9. 12. 15. 18. 21.]
  [ 4.  8. 12. 16. 20. 24. 28.]
  [ 5. 10. 15. 20. 25. 30. 35.]
  [ 6. 12. 18. 24. 30. 36. 42.]
  [ 7. 14. 21. 28. 35. 42. 49.]]]

[[[[ 24.  42.  60.  78.  96. 114.  60.]
   [ 48.  84. 120. 156. 192. 228. 120.]
   [ 72. 126. 180. 234. 288. 342. 180.]
   [ 96. 168. 240. 312. 384. 456. 240.]
   [120. 210. 300. 390. 480. 570. 300.]
   [144. 252. 360. 468. 576. 684. 360.]
   [104. 182. 260. 338. 416. 494. 260.]]

  [[ 24.  42.  60.  78.  96. 114.  60.]
   [ 48.  84. 120. 156. 192. 228. 120.]
   [ 72. 126. 180. 234. 288. 342. 180.]
   [ 96. 168. 240. 312. 384. 456. 240.]
   [120. 210. 300. 390. 480. 570. 300.]
   [144. 252. 360. 468. 576. 684. 360.]
   [104. 182. 260. 338. 416. 494. 260.]]

  [[ 24.  42.  60.  78.  96. 114.  60.]
   [ 48.  84. 120. 156. 192. 228. 120.]
   [ 72. 126. 180. 234. 288. 342. 180.]
   [ 96. 168. 240. 312. 384. 456. 240.]
   [120. 210. 300. 390. 480. 570. 300.]
   [144. 252. 360. 468. 576. 684. 360.]
   [104. 182. 260. 338. 416. 494. 260.]]]]
