#ifndef DIRECTSLAM_LOG_H
#define DIRECTSLAM_LOG_H

#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"

#include <iostream>
#include <memory>

#ifdef NO_LOG

#define Log_info(...)   ;
#define Log_debug(...)  ;
#define Log_warn(...)   ;
#define Log_error(...)  ;
#define Log_trace(...)  ;

#else

#if defined(__ANDROID__) 
#define Log_info(...)   { Log::console->info(__VA_ARGS__); }
#define Log_debug(...)  { Log::console->debug(__VA_ARGS__); }
#define Log_warn(...)   { Log::console->warn(__VA_ARGS__); }
#define Log_error(...)  { Log::console->error(__VA_ARGS__); }
#define Log_trace(...)  { Log::console->trace(__VA_ARGS__); }

class Log
{
    public:
        static std::shared_ptr<spdlog::logger> console;

        static void set_level(int level) {
            spdlog::level::level_enum e = static_cast<spdlog::level::level_enum>(level);
            console->set_level(e);
        };

        static void set_pattern(const std::string& pattern) {
            console->set_pattern(pattern);
        };

        static void set_writeFile(bool ifWriteFile) {}
};

#else

#define Log_info(...)   { Log::console->info(__VA_ARGS__); if (Log::writeFile) Log::filelog->info(__VA_ARGS__); }
#define Log_debug(...)  { Log::console->debug(__VA_ARGS__); if (Log::writeFile) Log::filelog->debug(__VA_ARGS__); }
#define Log_warn(...)   { Log::console->warn(__VA_ARGS__); if (Log::writeFile) Log::filelog->warn(__VA_ARGS__); }
#define Log_error(...)  { Log::console->error(__VA_ARGS__); if (Log::writeFile) Log::filelog->error(__VA_ARGS__); }
#define Log_trace(...)  { Log::console->trace(__VA_ARGS__); if (Log::writeFile) Log::filelog->trace(__VA_ARGS__); }

class Log
{
    public:
        static std::shared_ptr<spdlog::logger> console;
        static std::shared_ptr<spdlog::logger> filelog;

        static bool writeFile;

        static void set_level(int level) {
            spdlog::level::level_enum e = static_cast<spdlog::level::level_enum>(level);
            console->set_level(e);
            filelog->set_level(e);
        };

        static void set_pattern(const std::string& pattern) {
            console->set_pattern(pattern);
            filelog->set_pattern(pattern);
        };

        static void set_writeFile(bool ifWriteFile) {
            writeFile = ifWriteFile;
        }
};

#endif  // __ANDROID__

#endif  // NO_LOG

#endif  // DIRECTSLAM_LOG_H
