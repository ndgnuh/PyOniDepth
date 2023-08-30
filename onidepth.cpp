#define PY_SSIZE_T_CLEAN
#include "./onidepth_camera.cpp"
#include <OpenNI.h>
#include <Python.h>
#include <stdio.h>

int count = 0;
PyObject *testfn(PyObject *self, PyObject *args) {
  count += 1;
  printf("Count: %d\n", count);
  Py_RETURN_NONE;
}

static PyMethodDef onidepth_methods[] = {
    {"testfn", testfn, METH_VARARGS, "Test"},
    {"initialize", (PyCFunction)onidepth_initialize,
     METH_VARARGS | METH_KEYWORDS, "Initialize OpenNI and start capturing"},
    {"destroy", onidepth_destroy, METH_VARARGS,
     "Close OpenNI and stop capturing"},
    {"get_frame", onidepth_getframe, METH_VARARGS, "get the depth frame"},
    {"get_max_depth", onidepth_get_max_depth, METH_VARARGS, "get max depth"},
    {"get_min_depth", onidepth_get_min_depth, METH_VARARGS, "get min depth"},
    {"set_max_depth", onidepth_set_max_depth, METH_VARARGS, "set max depth"},
    {"set_min_depth", onidepth_set_min_depth, METH_VARARGS, "set min depth"},
    {NULL, NULL, 0, NULL} // Sentinel
};

PyMODINIT_FUNC PyInit_onidepth() {
  import_array();
  static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT, .m_name = "onidepth",
      .m_doc = "OpenNI depth camera", .m_size = -1, //
      onidepth_methods};

  PyObject *module = PyModule_Create(&moduledef);
  PyModule_AddIntMacro(module, PIXEL_FORMAT_DEPTH_1_MM);
  PyModule_AddIntMacro(module, PIXEL_FORMAT_DEPTH_100_UM);
  PyModule_AddIntMacro(module, PIXEL_FORMAT_SHIFT_9_2);
  PyModule_AddIntMacro(module, PIXEL_FORMAT_SHIFT_9_3);
  return module;
}
