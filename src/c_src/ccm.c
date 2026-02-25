#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

static void rgb_to_oklab(float *rgb, float *lab, int arr_len)
{
}

static void py_init_ccm_in_random_lab_space_impl(float *out_ccm_data,            // 6
                                                 float *rgb_mean_array,          // (18, 3)
                                                 float *linear_lab_patch_target, // (18, 3)
                                                 float *lab_patch_weight,        // (18, 3)
                                                 float *gamma_curve,             // (257)
                                                 int gamma_en)
{
    float oklab_18_patch[18 * 3] = {0.0F};
    rgb_to_oklab(rgb_mean_array, oklab_18_patch, 18 * 3); // (18, 3)
    // transpose (3, 18)
    float rgb_18_patch[18 * 3] = {0.0F}; // (3, 18)
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 18; j++)
        {
            rgb_18_patch[i * 18 + j] = rgb_mean_array[j * 3 + i];
        }
    }

    const float step_unit = 0.125F;
    float init_ccm[6] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    float best_lab_distance = 100000.F;
    int best_i = -1;
    int best_j = -1;
    int best_k = -1;
    int best_l = -1;
    int best_m = -1;
    int best_n = -1;

    for(int i=0; i<9; i++)
    {
        for(int j=0; j<9; j++)
        {
            for(int k=0; k<9; k++)
            {
                for(int l=0; l<9; l++)
                {
                    for(int m=0; m<9; m++)
                    {
                        for(int n=0; n<9; n++)
                        {
                            init_ccm[0] = i * step_unit;
                            init_ccm[1] = j * step_unit;
                            init_ccm[2] = k * step_unit;
                            init_ccm[3] = l * step_unit;
                            init_ccm[4] = m * step_unit;
                            init_ccm[5] = n * step_unit;
                        }
                    }
                }
            }
        }
    }
}

static PyObject *py_init_ccm_in_random_lab_space(PyObject *self, PyObject *args)
{
    PyArrayObject *rgb_mean_array = NULL;
    PyArrayObject *linear_lab_patch_target = NULL;
    PyArrayObject *lab_patch_weight = NULL;
    PyArrayObject *gamma_curve = NULL;
    PyObject *gamma_en = NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!:init_ccm_in_random_lab_space",
                          &PyArray_Type, &rgb_mean_array, &PyArray_Type, &linear_lab_patch_target,
                          &PyArray_Type, &lab_patch_weight, &PyBool_Type, &gamma_en,
                          &PyArray_Type, &gamma_curve))
    {
        return NULL;
    }

    int gamma_en_flag = (gamma_en == Py_True);

    npy_intp out_dims[1] = {6}; // ccm init len = 6
    PyArrayObject *out_ccm = (PyArrayObject *)PyArray_SimpleNew(1, out_dims, NPY_FLOAT32);
    float *out_ccm_data = (float *)PyArray_DATA(out_ccm);
    for (int i = 0; i < out_dims[0]; i++)
    {
        out_ccm_data[i] = 0.0F;
    }

    /* Check array properties */
    if (PyArray_NDIM(rgb_mean_array) != 2 || PyArray_NDIM(linear_lab_patch_target) != 2 ||
        PyArray_NDIM(lab_patch_weight) != 2 || PyArray_NDIM(gamma_curve) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "Input array shape error");
        return (PyObject *)out_ccm;
    }

    if (PyArray_TYPE(rgb_mean_array) != NPY_FLOAT32)
    {
        PyErr_SetString(PyExc_ValueError, "rgb_mean_array must be float type");
        return (PyObject *)out_ccm;
    }
    if (PyArray_TYPE(linear_lab_patch_target) != NPY_FLOAT32)
    {
        PyErr_SetString(PyExc_ValueError, "linear_lab_patch_target must be float type");
        return (PyObject *)out_ccm;
    }
    if (PyArray_TYPE(lab_patch_weight) != NPY_FLOAT32)
    {
        PyErr_SetString(PyExc_ValueError, "lab_patch_weight must be float type");
        return (PyObject *)out_ccm;
    }
    if (PyArray_TYPE(gamma_curve) != NPY_UINT8)
    {
        PyErr_SetString(PyExc_ValueError, "gamma_curve must be uint8 type");
        return (PyObject *)out_ccm;
    }

    /* Get dimensions */
    npy_intp *dims = PyArray_DIMS(rgb_mean_array);
    int patch_cnt = (int)dims[0];
    int channel_cnt = (int)dims[1];

    if (patch_cnt != 18 || channel_cnt != 3)
    {
        PyErr_SetString(PyExc_ValueError, "rgb_mean_array shape must be (18 * 3)");
        return (PyObject *)out_ccm;
    }
    if ((PyArray_DIMS(linear_lab_patch_target))[0] != 18 || (PyArray_DIMS(linear_lab_patch_target))[1] != 3)
    {
        PyErr_SetString(PyExc_ValueError, "linear_lab_patch_target shape must be (18 * 3)");
        return (PyObject *)out_ccm;
    }
    if ((PyArray_DIMS(lab_patch_weight))[0] != 18 || (PyArray_DIMS(lab_patch_weight))[1] != 3)
    {
        PyErr_SetString(PyExc_ValueError, "lab_patch_weight shape must be (18 * 3)");
        return (PyObject *)out_ccm;
    }
    if ((PyArray_DIMS(gamma_curve))[0] != 257)
    {
        PyErr_SetString(PyExc_ValueError, "gamma_curve shape must be (257, )");
        return (PyObject *)out_ccm;
    }

    py_init_ccm_in_random_lab_space_impl(out_ccm_data,
                                         (float *)PyArray_DATA(rgb_mean_array),
                                         (float *)PyArray_DATA(linear_lab_patch_target),
                                         (float *)PyArray_DATA(lab_patch_weight),
                                         (float *)PyArray_DATA(gamma_curve),
                                         gamma_en_flag);

    return (PyObject *)out_ccm;
}

/* Module method table */
static PyMethodDef ccm_accelerate_methods[] =
    {
        {"init_ccm_in_random_lab_space", py_init_ccm_in_random_lab_space, METH_VARARGS, "init ccm iter in lab space"},
        {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Module definition */
static struct PyModuleDef ccm_accelerate_module =
    {
        PyModuleDef_HEAD_INIT,
        "ccm_accelerate",                         /* name of module */
        "accelerate ccm optimize in C extension", /* module documentation */
        -1,                                       /* size of per-interpreter state or -1 */
        ccm_accelerate_methods};

/* Module initialization */
PyMODINIT_FUNC PyInit_ccm(void)
{
    import_array(); /* Initialize NumPy */
    return PyModule_Create(&ccm_accelerate_module);
}