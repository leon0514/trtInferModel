#ifndef LOGGER_HPP__
#define LOGGER_HPP__

#define INFO(...) __log_func(__FILE__, __LINE__, __VA_ARGS__)
void __log_func(const char *file, int line, const char *fmt, ...);

#endif