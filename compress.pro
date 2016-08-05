TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

QMAKE_CXXFLAGS += -pg -O1

INCLUDEPATH += .    \
    utils   \
    image

SOURCES += main.cpp \
    thirdparty/stb_image/stb_image_impl.c \
    image/sampler.cpp \
    image/transform.cpp \
    utils/utils.cpp

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
    image/sampler.h

DISTFILES += \
    ../tasks.todo
