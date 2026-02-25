#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <string.h>
// #include <stdio.h>

typedef enum 
{
    RGGB = 0,
    GRBG = 1,
    GBRG = 2,
    BGGR = 3,
} bayer_type_t;

typedef enum 
{
    PIX_R = 0,
    PIX_GR = 1,
    PIX_GB = 2,
    PIX_B = 3,
    PIX_Y = 4
} pixel_bayer_type;

static pixel_bayer_type type_RGGB[4] = { PIX_R, PIX_GR, PIX_GB, PIX_B };
static pixel_bayer_type type_GRBG[4] = { PIX_GR, PIX_R, PIX_B, PIX_GB };
static pixel_bayer_type type_GBRG[4] = { PIX_GB, PIX_B, PIX_R, PIX_GR };
static pixel_bayer_type type_BGGR[4] = { PIX_B, PIX_GB, PIX_GR, PIX_R };

static pixel_bayer_type get_pixel_bayer_type(uint32_t y, uint32_t x, bayer_type_t by)
{
    uint32_t pos = ((y & 0x1u) << 1) | (x & 0x1u);

    const pixel_bayer_type* type = type_RGGB;
    if (by == RGGB)
    {
        type = type_RGGB;
    }
    else if (by == GRBG)
    {
        type = type_GRBG;
    }
    else if (by == GBRG)
    {
        type = type_GBRG;
    }
    else if (by == BGGR)
    {
        type = type_BGGR;
    }

    return type[pos];
}

static uint8_t* demosaic_extend_board(const uint8_t* input_raw, uint32_t xsize, uint32_t ysize, uint32_t EXT_W, uint32_t EXT_H)
{
    if (input_raw == NULL)
    {
        return NULL;
    }

    uint8_t* dst = (uint8_t*)malloc((xsize + EXT_W) * (ysize + EXT_H));

    for (uint32_t y = EXT_H / 2; y < ysize + EXT_H / 2; y++)
    {
        for (uint32_t x = EXT_W / 2; x < xsize + EXT_W / 2; x++)
        {
            dst[y * (xsize + EXT_W) + x] = input_raw[(y - EXT_H / 2) * xsize + x - EXT_W / 2];
        }
    }
    // top
    for (uint32_t y = 0; y < EXT_H / 2; y++)
    {
        for (uint32_t x = EXT_W / 2; x < xsize + EXT_W / 2; x++)
        {
            dst[y * (xsize + EXT_W) + x] = dst[(EXT_H - y) * (xsize + EXT_W) + x];
        }
    }
    // bottom
    for (uint32_t y = ysize + EXT_H / 2; y < ysize + EXT_H; y++)
    {
        for (uint32_t x = EXT_W / 2; x < xsize + EXT_W / 2; x++)
        {
            dst[y * (xsize + EXT_W) + x] = dst[(2 * ysize + EXT_H - 2 - y) * (xsize + EXT_W) + x];
        }
    }
    // left
    for (uint32_t y = 0; y < ysize + EXT_H; y++)
    {
        for (uint32_t x = 0; x < EXT_W / 2; x++)
        {
            dst[y * (xsize + EXT_W) + x] = dst[y * (xsize + EXT_W) + (EXT_W - x)];
        }
    }
    // right
    for (uint32_t y = 0; y < ysize + EXT_H; y++)
    {
        for (uint32_t x = xsize + EXT_W / 2; x < xsize + EXT_W; x++)
        {
            dst[y * (xsize + EXT_W) + x] = dst[y * (xsize + EXT_W) + (2 * xsize + EXT_W - 2 - x)];
        }
    }

    return dst;
}

#if (defined __GNUC__)
__attribute__((always_inline))
#elif (defined _MSC_VER)
__forceinline
#endif
static int32_t clip3(int32_t x, int32_t min_, int32_t max_)
{
    return (x > max_) ? max_ : ((x < min_) ? min_ : x);
}

void demosaic_bayer_bilinear(const uint8_t* indata, uint8_t* outdata, uint32_t width, uint32_t height, bayer_type_t bayer)
{
    pixel_bayer_type pix_type;
    uint32_t x_, y_;
    uint32_t r_0, r_1, r_2, r_3, r_c;
    uint8_t g_0, g_1, g_2, g_3, g_4, g_5, g_6, g_7, g_c;
    uint8_t b_0, b_1, b_2, b_3, b_c;
    uint32_t xsize = width;
    uint32_t ysize = height;
    uint32_t EXT_X = 8;
    uint32_t EXT_Y = 8;
    const int32_t max_val = 255;

    uint8_t* extend = demosaic_extend_board(indata, xsize, ysize, EXT_X, EXT_Y);

    for (uint32_t row = 0; row < ysize; row++)
    {
        for (uint32_t col = 0; col < xsize; col++)
        {
            pix_type = get_pixel_bayer_type(row, col, bayer);
            y_ = row + EXT_Y / 2;
            x_ = col + EXT_X / 2;
            if (pix_type == PIX_R)
            {
                outdata[row * xsize * 3 + col * 3 + 2] = extend[y_ * (xsize + EXT_X) + x_];
                g_0 = extend[y_ * (xsize + EXT_X) + x_ - 1];
                g_1 = extend[y_ * (xsize + EXT_X) + x_ + 1];
                g_2 = extend[(y_ - 1) * (xsize + EXT_X) + x_];
                g_3 = extend[(y_ + 1) * (xsize + EXT_X) + x_];
                r_c = extend[y_ * (xsize + EXT_X) + x_];
                r_0 = extend[(y_ - 2) * (xsize + EXT_X) + x_];
                r_1 = extend[(y_ + 2) * (xsize + EXT_X) + x_];
                r_2 = extend[y_ * (xsize + EXT_X) + x_ - 2];
                r_3 = extend[y_ * (xsize + EXT_X) + x_ + 2];

                int32_t g_out_val = (2 * (int32_t)(g_0 + g_1 + g_2 + g_3) + 4 * (int32_t)(r_c) - (int32_t)(r_0 + r_1 + r_2 + r_3) + 4) >> 3;
                outdata[row * xsize * 3 + col * 3 + 1] = (uint8_t)clip3(g_out_val, 0, max_val);

                b_0 = extend[(y_ - 1) * (xsize + EXT_X) + x_ - 1];
                b_1 = extend[(y_ - 1) * (xsize + EXT_X) + x_ + 1];
                b_2 = extend[(y_ + 1) * (xsize + EXT_X) + x_ - 1];
                b_3 = extend[(y_ + 1) * (xsize + EXT_X) + x_ + 1];
                int32_t b_out_val = (4 * (int32_t)(b_0 + b_1 + b_2 + b_3) + 12 * (int32_t)(r_c) - 3 * (int32_t)(r_0 + r_1 + r_2 + r_3) + 8) >> 4;
                outdata[row * xsize * 3 + col * 3 + 0] = (uint8_t)clip3(b_out_val, 0, max_val);
            }
            else if (pix_type == PIX_GR)
            {
                outdata[row * xsize * 3 + col * 3 + 1] = extend[y_ * (xsize + EXT_X) + x_];
                r_0 = extend[y_ * (xsize + EXT_X) + x_ - 1];
                r_1 = extend[y_ * (xsize + EXT_X) + x_ + 1];

                g_c = extend[y_ * (xsize + EXT_X) + x_];
                g_0 = extend[(y_ - 2) * (xsize + EXT_X) + x_];
                g_1 = extend[(y_ + 2) * (xsize + EXT_X) + x_];
                g_2 = extend[y_ * (xsize + EXT_X) + x_ - 2];
                g_3 = extend[y_ * (xsize + EXT_X) + x_ + 2];
                g_4 = extend[(y_ - 1) * (xsize + EXT_X) + x_ - 1];
                g_5 = extend[(y_ - 1) * (xsize + EXT_X) + x_ + 1];
                g_6 = extend[(y_ + 1) * (xsize + EXT_X) + x_ - 1];
                g_7 = extend[(y_ + 1) * (xsize + EXT_X) + x_ + 1];
                int32_t r_out_val = (8 * (int32_t)(r_0 + r_1) + 10 * (int32_t)(g_c) + (int32_t)(g_0) + (int32_t)(g_1) - 2 * (int32_t)(g_2 + g_3 + g_4 + g_5 + g_6 + g_7) + 8) >> 4;
                outdata[row * xsize * 3 + col * 3 + 2] = (uint8_t)clip3(r_out_val, 0, max_val);

                b_0 = extend[(y_ - 1) * (xsize + EXT_X) + x_];
                b_1 = extend[(y_ + 1) * (xsize + EXT_X) + x_];

                int32_t b_out_val = (8 * (int32_t)(b_0 + b_1) + 10 * (int32_t)(g_c) + (int32_t)(g_2) + (int32_t)(g_3) - 2 * (int32_t)(g_0 + g_1 + g_4 + g_5 + g_6 + g_7) + 8) >> 4;
                outdata[row * xsize * 3 + col * 3 + 0] = (uint8_t)clip3(b_out_val, 0, max_val);
            }
            else if (pix_type == PIX_GB)
            {
                outdata[row * xsize * 3 + col * 3 + 1] = extend[y_ * (xsize + EXT_X) + x_];
                b_0 = extend[y_ * (xsize + EXT_X) + x_ - 1];
                b_1 = extend[y_ * (xsize + EXT_X) + x_ + 1];

                g_c = extend[y_ * (xsize + EXT_X) + x_];
                g_0 = extend[(y_ - 2) * (xsize + EXT_X) + x_];
                g_1 = extend[(y_ + 2) * (xsize + EXT_X) + x_];
                g_2 = extend[y_ * (xsize + EXT_X) + x_ - 2];
                g_3 = extend[y_ * (xsize + EXT_X) + x_ + 2];
                g_4 = extend[(y_ - 1) * (xsize + EXT_X) + x_ - 1];
                g_5 = extend[(y_ - 1) * (xsize + EXT_X) + x_ + 1];
                g_6 = extend[(y_ + 1) * (xsize + EXT_X) + x_ - 1];
                g_7 = extend[(y_ + 1) * (xsize + EXT_X) + x_ + 1];

                int32_t b_out_val = (8 * (int32_t)(b_0 + b_1 + 1) + 10 * (int32_t)(g_c) + (int32_t)(g_0) + (int32_t)(g_1) - 2 * (int32_t)(g_2 + g_3 + g_4 + g_5 + g_6 + g_7) + 8) >> 4;
                outdata[row * xsize * 3 + col * 3 + 0] = (uint8_t)clip3(b_out_val, 0, max_val);

                r_0 = extend[(y_ - 1) * (xsize + EXT_X) + x_];
                r_1 = extend[(y_ + 1) * (xsize + EXT_X) + x_];
                int32_t r_out_val = (8 * (int32_t)(r_0 + r_1) + 10 * (int32_t)(g_c) + (int32_t)(g_2) + (int32_t)(g_3) - 2 * (int32_t)(g_0 + g_1 + g_4 + g_5 + g_6 + g_7) + 8) >> 4;
                outdata[row * xsize * 3 + col * 3 + 2] = (uint8_t)clip3(r_out_val, 0, max_val);
            }
            else if (pix_type == PIX_B)
            {
                outdata[row * xsize * 3 + col * 3 + 0] = extend[y_ * (xsize + EXT_X) + x_];
                g_0 = extend[y_ * (xsize + EXT_X) + x_ - 1];
                g_1 = extend[y_ * (xsize + EXT_X) + x_ + 1];
                g_2 = extend[(y_ - 1) * (xsize + EXT_X) + x_];
                g_3 = extend[(y_ + 1) * (xsize + EXT_X) + x_];

                b_c = extend[y_ * (xsize + EXT_X) + x_];
                b_0 = extend[(y_ - 2) * (xsize + EXT_X) + x_];
                b_1 = extend[(y_ + 2) * (xsize + EXT_X) + x_];
                b_2 = extend[y_ * (xsize + EXT_X) + x_ - 2];
                b_3 = extend[y_ * (xsize + EXT_X) + x_ + 2];

                int32_t g_out_val = (2 * (int32_t)(g_0 + g_1 + g_2 + g_3) + 4 * (int32_t)(b_c) - (int32_t)(b_0 + b_1 + b_2 + b_3) + 4) >> 3;
                outdata[row * xsize * 3 + col * 3 + 1] = (uint8_t)clip3(g_out_val, 0, max_val);

                r_0 = extend[(y_ - 1) * (xsize + EXT_X) + x_ - 1];
                r_1 = extend[(y_ - 1) * (xsize + EXT_X) + x_ + 1];
                r_2 = extend[(y_ + 1) * (xsize + EXT_X) + x_ - 1];
                r_3 = extend[(y_ + 1) * (xsize + EXT_X) + x_ + 1];

                int32_t r_out_val = (4 * (int32_t)(r_0 + r_1 + r_2 + r_3) + 12 * (int32_t)(b_c) - 3 * (int32_t)(b_0 + b_1 + b_2 + b_3) + 8) >> 4;
                outdata[row * xsize * 3 + col * 3 + 2] = (uint8_t)clip3(r_out_val, 0, max_val);
            }
        }
    }
    free(extend);
}


/* Python function: bayer2bgr */
static PyObject* py_bayer2bgr(PyObject* self, PyObject* args) {
    PyArrayObject *bayer_array = NULL;
    const char *pattern_str = NULL;
    PyObject *result = NULL;
    
    /* Parse arguments: numpy array and pattern string */
    if (!PyArg_ParseTuple(args, "O!s:bayer2bgr", 
                         &PyArray_Type, &bayer_array, 
                         &pattern_str)) {
        return NULL;
    }
    
    /* Check array properties */
    if (PyArray_NDIM(bayer_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Input must be a 2D array");
        return NULL;
    }
    
    if (PyArray_TYPE(bayer_array) != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError, "Input must be uint8 type");
        return NULL;
    }
    
    /* Get dimensions */
    npy_intp *dims = PyArray_DIMS(bayer_array);
    int height = (int)dims[0];
    int width = (int)dims[1];
    
    /* Parse pattern string */
    bayer_type_t pattern;
    if (strcmp(pattern_str, "RGGB") == 0) {
        pattern = RGGB;
    } else if (strcmp(pattern_str, "GRBG") == 0) {
        pattern = GRBG;
    } else if (strcmp(pattern_str, "GBRG") == 0) {
        pattern = GBRG;
    } else if (strcmp(pattern_str, "BGGR") == 0) {
        pattern = BGGR;
    } else {
        PyErr_SetString(PyExc_ValueError, 
                       "Pattern must be one of: RGGB, GRBG, GBRG, BGGR");
        return NULL;
    }
    
    /* Create output array (height, width, 3) */
    npy_intp out_dims[3] = {height, width, 3};
    PyArrayObject *out_array = (PyArrayObject*)PyArray_SimpleNew(
        3, out_dims, NPY_UINT8);
    
    if (!out_array) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output array");
        return NULL;
    }
    
    /* Get data pointers */
    uint8_t *bayer_data = (uint8_t*)PyArray_DATA(bayer_array);
    uint8_t *bgr_data = (uint8_t*)PyArray_DATA(out_array);
    
    /* Perform demosaicing */
    // printf("width %d, height %d, pattern %d\n", width, height, pattern);
    demosaic_bayer_bilinear(bayer_data, bgr_data, width, height, pattern);
    
    result = (PyObject*)out_array;
    return result;
}

/* Module method table */
static PyMethodDef DemosaicMethods[] = {
    {"bayer2bgr", py_bayer2bgr, METH_VARARGS, 
     "Convert Bayer pattern image to BGR using bilinear interpolation.\n"
     "Args:\n"
     "  bayer_array: 2D numpy uint8 array\n"
     "  pattern: One of 'RGGB', 'GRBG', 'GBRG', 'BGGR'\n"
     "Returns:\n"
     "  3D numpy uint8 array in BGR format"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Module definition */
static struct PyModuleDef demosaicmodule = {
    PyModuleDef_HEAD_INIT,
    "demosaic",  /* name of module */
    "Minimal Bayer demosaicing C extension",  /* module documentation */
    -1,  /* size of per-interpreter state or -1 */
    DemosaicMethods
};

/* Module initialization */
PyMODINIT_FUNC PyInit_demosaic(void) {
    import_array();  /* Initialize NumPy */
    return PyModule_Create(&demosaicmodule);
}