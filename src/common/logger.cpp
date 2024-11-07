#include "common/logger.hpp"
#include <string>
#include <algorithm>
#include <stdarg.h>

static std::string file_name(const std::string &path, bool include_suffix) 
{
    if (path.empty()) return "";

    int p = path.rfind('/');
    int e = path.rfind('\\');
    p = std::max(p, e);
    p += 1;

    // include suffix
    if (include_suffix) return path.substr(p);

    int u = path.rfind('.');
    if (u == -1) return path.substr(p);

    if (u <= p) u = path.size();
    return path.substr(p, u - p);
}

void __log_func(const char *file, int line, const char *fmt, ...) 
{
    va_list vl;
    va_start(vl, fmt);
    char buffer[2048];
    std::string filename = file_name(file, true);
    int n = snprintf(buffer, sizeof(buffer), "[%s:%d]: ", filename.c_str(), line);
    vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
    fprintf(stdout, "%s\n", buffer);
}