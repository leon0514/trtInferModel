#ifndef TIMER_HPP__
#define TIMER_HPP__

class Timer {
 public:
  Timer();
  virtual ~Timer();
  void start(void *stream = nullptr);
  float stop(const char *prefix = "Timer", bool print = true);

 private:
  void *start_, *stop_;
  void *stream_;
};

#endif