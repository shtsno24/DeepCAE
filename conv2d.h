
typedef struct{
    uint16_t height, width, depth;
}ARRAY_PARAMS_3D;

uint8_t conv2d(uint16_t, ARRAY_PARAMS_3D, int16_t*,
ARRAY_PARAMS_3D, int16_t* , uint16_t, uint16_t,
ARRAY_PARAMS_3D, int16_t*,
ARRAY_PARAMS_3D, int16_t*);
