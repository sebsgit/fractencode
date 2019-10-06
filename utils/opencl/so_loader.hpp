#pragma once

#ifdef _WIN32
    #include <windows.h>
#elif defined __linux__
    #include <dlfcn.h>
#else
    #error Unsupported platform !
#endif

#include <string>
#include <functional>

namespace so_loader {
namespace priv {
    #ifdef _WIN32
    using handle = HMODULE;
#elif defined __linux__
    using handle = void*;
#endif

    static auto load(const std::string& path){
#ifdef _WIN32
    return LoadLibraryA(path.c_str());
#elif defined __linux__
    dlerror();
    return dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
}

static auto get_symbol(handle h, const std::string& sym){
#ifdef _WIN32
    return GetProcAddress(h, sym.c_str());
#elif defined __linux__
    dlerror();
    return dlsym(h, sym.c_str());
#endif
}

static const char * get_error() {
    #ifdef _WIN32
        static char buff[256];
        snprintf(buff,256,"ERROR CODE: %lu\n", GetLastError());
        return buff;
    #elif defined __linux__
        return dlerror();
    #endif
}

static void close_handle(handle h){
    if (h){
    #ifdef _WIN32
        FreeLibrary(h);
    #elif defined __linux__
        dlclose(h);
    #endif
    }
}
}

class library {
public:
    library() noexcept = default;
    explicit library(const std::string& path)
        : d_(priv::load(path))
    {
    }
    library(library&& other) noexcept
        : d_(other.d_)
    {
        other.d_ = nullptr;
    }
    ~library() {
        this->close();
    }

    library& operator= (library&& other) noexcept {
        if (this->d_ != other.d_) {
            priv::close_handle(d_);
            d_ = other.d_;
            other.d_ = nullptr;
        }
        return *this;
    }

    library(const library&) = delete;
    library& operator= (const library&) = delete;

    bool is_open() const noexcept { return d_ != nullptr; }
    void close() {
        priv::close_handle(d_);
        d_ = nullptr;
    }
    bool has_function(const std::string& name) const {
        return priv::get_symbol(d_, name) != nullptr;
    }
    std::string error_string() const { return priv::get_error(); }
    template <typename FuncType>
    std::function<FuncType> function(const std::string& name) const {
        return reinterpret_cast<FuncType*>(priv::get_symbol(d_, name));
    }
    auto symbol(const std::string& name) const {
        return priv::get_symbol(d_, name);
    }
private:
    priv::handle d_ = nullptr;
};
}
