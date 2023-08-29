PYTHON_VERSION=3.11
.PHONY: build main lib

OPENCV_FLAGS := $(shell pkg-config --cflags --libs opencv4)
OPENNI_FLAGS := -I./sdk/Include/ -L./sdk/libs/ -lOpenNI2
PYTHON_FLAGS := $(shell  python3-config --cflags )
NUMPY_FLAGS := -I/usr/lib/python3/dist-packages/numpy/core/include/
CFLAGS=$(OPENNI_FLAGS) $(NUMPY_FLAGS) $(PYTHON_FLAGS)
INC_DIRS = sdk/Include/
USED_LIBS += OpenNI2

build:
	gcc -I /usr/include/python$(PYTHON_VERSION)/ \
		-I /usr/lib/python3/dist-packages/numpy/core/include/numpy/ \
		-I /usr/local/include/Linux-x86/ \
		test.c -shared -o libtest.so 

lib:
	g++ onidepth.cpp -shared -o onidepth.so $(PYTHON_FLAGS) $(OPENNI_FLAGS) -fPIC
	# python test.py

main: 
	g++ main.cpp -o main -Werror $(CFLAGS)
		
