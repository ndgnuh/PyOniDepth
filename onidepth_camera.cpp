#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define SAMPLE_READ_WAIT_TIMEOUT 2000
#include <OpenNI.h>
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/numpyconfig.h>

using namespace openni;

Device device;
VideoStream depth;
VideoFrameRef frame;
/* int VideoStream::getMinPixelValue() */
/* int VideoStream::getMaxPixelValue() */
uint16_t minDepth = (uint16_t)depth.getMinPixelValue();
uint16_t maxDepth = (uint16_t)depth.getMaxPixelValue();

static PyObject *onidepth_initialize(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  /* Parse args */
  int width = 0;
  int height = 0;
  int fps = 0;
  int pixel_format = 0;
  static char *kwlist[] = {"width", "height", "fps", "pixel_format", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiii", kwlist, &width,
                                   &height, &fps, &pixel_format)) {
    return NULL;
  }

  Status rc;

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
  width = width ? width : mode.getResolutionX();
  height = height ? height : mode.getResolutionX();
  fps = fps ? fps : mode.getFps();
  pixel_format = pixel_format ? pixel_format : mode.getPixelFormat();
  mode.setResolution(width, height);
  mode.setFps(fps);
  mode.setPixelFormat((PixelFormat)pixel_format);
  rc = depth.setVideoMode(mode);
  if (rc != STATUS_OK) {
    printf("Fail to set video mode. Details:\n%s", OpenNI::getExtendedError());
    Py_RETURN_FALSE;
  }

  // show the video mode
  printf("[init] Mode %dx%d@%d - pixel format: %d\n", width, height, fps,
         pixel_format);

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

  // change min/max depth
  minDepth = (uint16_t)depth.getMinPixelValue();
  maxDepth = (uint16_t)depth.getMaxPixelValue();
  printf("Min depth: %d - Max depth %d\n", minDepth, maxDepth);

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

  // correction of depth data, switch to higher -> closest
  // Normalize depth data also
  const npy_intp dims[2] = {height, width};
  uint16_t depthRange = maxDepth - minDepth;
  uint16_t *depthData = (uint16_t *)frame.getData();
  size_t size = frame.getDataSize();
  float normDepthData[size];

  for (int i = 0; i < width * height; i++) {
    if (depthData[i] >= maxDepth)
      depthData[i] = (uint16_t)1;
    else if (depthData[i] <= minDepth)
      depthData[i] = (uint16_t)0;
    else
      depthData[i] = maxDepth - (depthData[i] - minDepth);
    float px = (depthData[i]);
    normDepthData[i] = ((float)px - minDepth) / depthRange;
  }

  // convert to numpy array
  PyObject *pyframe =
      PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, (void *)normDepthData);

  return pyframe;
}

/*
 * get minDepth
 */
PyObject *onidepth_get_min_depth(PyObject *self, PyObject *arg) {
  return PyLong_FromLong((long)minDepth);
}

/*
 * get maxDepth
 */
PyObject *onidepth_get_max_depth(PyObject *self, PyObject *arg) {
  return PyLong_FromLong((long)maxDepth);
}

/*
 * set minDepth
 */
PyObject *onidepth_set_min_depth(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, "i", &minDepth)) {
    Py_RETURN_FALSE;
  }
  Py_RETURN_TRUE;
}

/*
 * set maxDepth
 */
PyObject *onidepth_set_max_depth(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, "i", &maxDepth)) {
    Py_RETURN_FALSE;
  }
  Py_RETURN_TRUE;
}
