#define PY_SSIZE_T_CLEAN
#include "./onidepth_camera.cpp"
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
    {"initialize", onidepth_initialize, METH_VARARGS,
     "Initialize OpenNI and start capturing"},
    {"destroy", onidepth_destroy, METH_VARARGS,
     "Close OpenNI and stop capturing"},
    {"get_frame", onidepth_getframe, METH_VARARGS, "get the depth frame"},
    {NULL, NULL, 0, NULL} // Sentinel
};

PyMODINIT_FUNC PyInit_onidepth() {
  static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT, .m_name = "onidepth",
      .m_doc = "OpenNI depth camera", .m_size = -1, //
      onidepth_methods};

  PyObject *module = PyModule_Create(&moduledef);
  return module;
}
