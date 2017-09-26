#include "Log.h"

#ifdef NO_LOG

#else

#if defined(__ANDROID__)

std::shared_ptr<spdlog::logger> Log::console = spdlog::android_logger("android", "TinySLAMNative");

#else

std::shared_ptr<spdlog::logger> Log::console = spdlog::stdout_color_mt("console");
std::shared_ptr<spdlog::logger> Log::filelog = spdlog::basic_logger_mt("filelog", "logfile.txt");;
bool Log::writeFile = false;

#endif

#endif
