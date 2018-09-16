#pragma once

#define FRAC_NO_COPY(Klass) Klass(const Klass&) = delete; \
    Klass& operator=(const Klass&) = delete
