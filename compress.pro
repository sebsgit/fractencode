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

INCLUDEPATH += .	\
        utils   \
        image   \
        process	\
        schedule \
        thirdparty/gsl/include/

SOURCES += main.cpp \
    encode/Classifier2.cpp \
    encode/EncodingEngine.cpp \
    encode/EncodingEngine2.cpp \
    image/ImageIO.cpp \
    image/ImageStatistics.cpp \
        thirdparty/stb_image/stb_image_impl.c \
        image/sampler.cpp \
        image/transform.cpp \
    utils/sse_utils.cpp \
        utils/utils.cpp \
		utils/sse_debug.cpp \
        image/imageutils.cpp \
        image/partition/gridpartition.cpp \
        image/partition/quadtreepartition.cpp \
        process/gaussian5x5.cpp \
        process/sobel.cpp \
        encode/edgeclassifier.cpp

HEADERS += \
    image/Image2.hpp \
    image/ImageIO.hpp \
    image/ImageStatistics.hpp \
    image/partition2.hpp \
        thirdparty/stb_image/stb_image.h \
        thirdparty/stb_image/stb_image_write.h \
        utils/buffer.hpp	\
        image/image.h   \
        image/transform.h \
        utils/size.hpp \
        image/metrics.h \
        utils/point2d.hpp \
        image/partition.h \
        encode/encoder.h \
        image/sampler.h \
        encode/classifier.h \
        encode/transformmatcher.h \
		encode/TransformEstimator.h \
        image/partition/gridpartition.h \
        encode/datatypes.h \
        utils/timer.h \
        utils/sse_debug.h \
        image/partition/quadtreepartition.h \
        process/gaussian5x5.h \
        process/abstractprocessor.h \
        process/sobel.h \
        encode/edgeclassifier.h

OBJECTS_DIR = build

DEFINES += FRAC_WITH_AVX

QMAKE_CXXFLAGS_DEBUG *= -pg
QMAKE_LFLAGS_DEBUG *= -pg
