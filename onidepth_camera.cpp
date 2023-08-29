// 2000ms
#define SAMPLE_READ_WAIT_TIMEOUT 2000
#include <OpenNI.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/numpyconfig.h>

using namespace openni;

Device device;
VideoStream depth;
VideoFrameRef frame;

static PyObject *onidepth_initialize(PyObject *self, PyObject *args) {
  Status rc;
  import_array();

  // initialize OpenNI
  rc = OpenNI::initialize();
  if (rc != STATUS_OK) {
    printf("Failed to init OpenNI. Details: \n%s", OpenNI::getExtendedError());
    Py_RETURN_FALSE;
  }

  // open device
  rc = device.open(ANY_DEVICE);
  if (rc != STATUS_OK) {
    printf("Failed to open camera. Details: \n%s", OpenNI::getExtendedError());
    Py_RETURN_FALSE;
  }

  // create depth stream
  rc = depth.create(device, SENSOR_DEPTH);
  if (rc != STATUS_OK) {
    printf("Failed to create depth stream. Details:\n%s",
           OpenNI::getExtendedError());
    Py_RETURN_FALSE;
  }

  // set video mode

  VideoMode mode = depth.getVideoMode();
  // 640x400, not 640x480
  printf("Mode: %d %d", mode.getResolutionX(), mode.getResolutionY());
  mode.setResolution(640, 400);
  mode.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
  mode.setFps(30);
  rc = depth.setVideoMode(mode);
  if (rc != STATUS_OK) {
    printf("Fail to set video mode. Details:\n%s", OpenNI::getExtendedError());
  }

  // start the depth stream
  rc = depth.start();
  if (rc != STATUS_OK) {
    printf("Failed to start depth stream. Details:\n%s",
           OpenNI::getExtendedError());
    Py_RETURN_FALSE;
  }

  // wait for it to start
  bool waitOk = false;
  VideoStream *pDepthStream = &depth;
  for (int i = 0; i < 10; i++) {
    int changedStreamDummy;
    rc = OpenNI::waitForAnyStream(&pDepthStream, 1, &changedStreamDummy,
                                  SAMPLE_READ_WAIT_TIMEOUT);
    if (rc == STATUS_OK) {
      waitOk = true;
      break;
    } else {
      printf("Depth stream timeout, trying again\n");
    }
  }

  device.setImageRegistrationMode(
      ImageRegistrationMode::IMAGE_REGISTRATION_DEPTH_TO_COLOR);

  // timeout
  if (!waitOk) {
    printf("Depth stream timeout. Details:\n%s", OpenNI::getExtendedError());
    Py_RETURN_FALSE;
  }

  Py_RETURN_TRUE;
}

PyObject *onidepth_destroy(PyObject *self, PyObject *args) {
  depth.stop();
  depth.destroy();
  device.close();
  OpenNI::shutdown();
  Py_RETURN_FALSE;
}

PyObject *onidepth_getframe(PyObject *self, PyObject *args) {
  int width, height;

  depth.readFrame(&frame);
  if (!frame.isValid()) {
    Py_RETURN_NONE;
  }

  width = frame.getWidth();
  height = frame.getHeight();
  const npy_intp dims[2] = {height, width};
  PyObject *pyframe =
      PyArray_SimpleNewFromData(2, dims, NPY_UINT16, (void *)frame.getData());

  return pyframe;
}
