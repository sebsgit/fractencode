TEMPLATE = app
TARGET = compress
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += debug_and_release

INCLUDEPATH += .    \
    utils   \
    image

SOURCES += main.cpp \
    thirdparty/stb_image/stb_image_impl.c \
    image/sampler.cpp \
    image/transform.cpp \
    utils/utils.cpp \
    image/imageutils.cpp \
    image/partition/gridpartition.cpp

HEADERS += \
    thirdparty/stb_image/stb_image.h \
    thirdparty/stb_image/stb_image_write.h \
    utils/buffer.hpp    \
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
    image/partition/gridpartition.h \
    encode/datatypes.h \
    utils/timer.h

DISTFILES += \
    ../tasks.todo

OBJECTS_DIR = build


QMAKE_CXXFLAGS_DEBUG *= -pg
QMAKE_LFLAGS_DEBUG *= -pg
