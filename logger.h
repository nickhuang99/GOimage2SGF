#ifndef LOGGER_H
#define LOGGER_H

#include <chrono>  // For system_clock
#include <cstring> // For strrchr
#include <filesystem> // For C++17 directory creation (optional, can use POSIX mkdir)
#include <fstream>
#include <iomanip> // For std::put_time, std::setfill, std::setw
#include <mutex>   // For thread safety (optional for now, but good to consider)
#include <sstream>
#include <string>

// Log Levels
enum class LogLevel {
  NONE = 0,    // No logging
  ERROR = 1,   // Critical errors
  WARNING = 2, // Warnings
  INFO = 3,    // General information (Default)
  DEBUG = 4    // Detailed debug information
};

class Logger {
public:
  // Constructor: Called by LOG_XXX macros, captures message level and location
  Logger(LogLevel message_level, const char *file, int line,
         const char *function_name);
  // Destructor: Ensures the buffered message is flushed to the log file
  ~Logger();

  // Static methods to control global logger behavior
  static void setGlobalLogLevel(LogLevel level);
  static LogLevel getGlobalLogLevel();
  static void setLogFile(const std::string &file_path);
  static void init(const std::string &initial_log_path = "share/log.txt",
                   LogLevel initial_level = LogLevel::INFO); // Explicit init

  // Stream operator to append message parts
  template <typename T> Logger &operator<<(const T &msg) {
    if (m_should_log_this_message) {
      m_buffer << msg;
    }
    return *this;
  }

  // Handle std::endl and other ostream manipulators (like std::flush)
  Logger &operator<<(std::ostream &(*manip)(std::ostream &));

private:
  std::ostringstream
      m_buffer; // Buffer for the current message (prefix + user parts)
  LogLevel m_message_level_instance; // Level of THIS specific message
  bool m_should_log_this_message; // True if this message's level is <= global
                                  // level
  bool m_prefix_and_timestamp_generated; // To ensure prefix is added only once
                                         // per message

  // Static members for global state and file stream
  static std::ofstream s_log_file_stream;
  static LogLevel s_global_log_level;
  static std::string s_log_file_path;
  static bool
      s_is_initialized_flag; // Tracks if static init (file open) has occurred
  static std::mutex s_log_mutex; // For thread-safe file writing

  // Static helper to write the complete, formatted string to file
  // (synchronized)
  static void writeStringToFile(const std::string &message);
  // Helper to generate prefix (called by constructor if should_log)
  void generatePrefix(const char *file, int line, const char *function_name);
};

// Macros to create temporary Logger instances
#define LOG_ERROR                                                              \
  Logger(LogLevel::ERROR, __FILE__, __LINE__, __PRETTY_FUNCTION__)
#define LOG_WARN                                                               \
  Logger(LogLevel::WARNING, __FILE__, __LINE__, __PRETTY_FUNCTION__)
#define LOG_INFO Logger(LogLevel::INFO, __FILE__, __LINE__, __PRETTY_FUNCTION__)
#define LOG_DEBUG                                                              \
  Logger(LogLevel::DEBUG, __FILE__, __LINE__, __PRETTY_FUNCTION__)

// For messages that should always print to console (e.g., help or pre-init
// errors)
#define CONSOLE_OUT std::cout
#define CONSOLE_ERR std::cerr

#endif // LOGGER_H