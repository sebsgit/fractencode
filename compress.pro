TEMPLATE = app
TARGET = compress
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += debug_and_release

!win32 {
    QMAKE_CXXFLAGS += -msse -msse2 -mavx -mavx2 -march=native
}

unix {
    QMAKE_LFLAGS += -pthread
}

DEFINES += FRAC_TESTS

INCLUDEPATH += .	\
        utils   \
        image   \
        process	\
        schedule \
        thirdparty/gsl/include/

SOURCES += main.cpp \
    encode/Classifier2.cpp \
    encode/EncodingEngine2.cpp \
    image/ImageIO.cpp \
    image/ImageStatistics.cpp \
    tests/ClassifierTest.cpp \
    tests/CodebookGeneratorTests.cpp \
    tests/ImageIOTest.cpp \
    tests/ImageSamplerTest.cpp \
    tests/ImageStatisticsTest.cpp \
    tests/PartitionTests.cpp \
    tests/TransformEstimatorTest.cpp \
    tests/TransformMatcherTest.cpp \
    thirdparty/stb_image/stb_image_impl.c \
    image/transform.cpp \
    utils/sse_utils.cpp \
    utils/utils.cpp \
    utils/sse_debug.cpp

HEADERS += \
    encode/Classifier2.hpp \
    encode/DecodeUtils.hpp \
    encode/Encoder2.hpp \
    encode/EncodingEngine2.hpp \
    encode/Quantizer.hpp \
    encode/TransformEstimator2.hpp \
    encode/encode_parameters.h \
    gpu/cuda/CudaConf.h \
    gpu/cuda/CudaEncoderBackend.h \
    gpu/cuda/CudaEncodingEngine.h \
    gpu/cuda/CudaPointer.h \
    gpu/opencl/OpenCLEncodingEngine.hpp \
    image/Image2.hpp \
    image/ImageIO.hpp \
    image/ImageStatistics.hpp \
    image/partition2.hpp \
    tests/catch.hpp \
    thirdparty/stb_image/stb_image.h \
    thirdparty/stb_image/stb_image_write.h \
    utils/buffer.hpp	\
    image/transform.h \
    utils/size.hpp \
    image/metrics.h \
    utils/point2d.hpp \
    image/sampler.h \
    encode/transformmatcher.h \
    encode/TransformEstimator.h \
    encode/datatypes.h \
    utils/timer.h \
    utils/sse_debug.h \
    process/abstractprocessor.h

OBJECTS_DIR = build

DEFINES += FRAC_WITH_AVX

QMAKE_CXXFLAGS_DEBUG *= -pg
QMAKE_LFLAGS_DEBUG *= -pg
