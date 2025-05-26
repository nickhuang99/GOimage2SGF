#include "logger.h"
#include <iostream> // For CONSOLE_ERR during init failure

// Static member definitions
std::ofstream Logger::s_log_file_stream;
LogLevel Logger::s_global_log_level =
    LogLevel::INFO; // Default global log level, overwritten by init()
std::string Logger::s_log_file_path =
    "share/log.txt"; // Default log file path, overwritten by init()
bool Logger::s_is_initialized_flag = false;
std::mutex Logger::s_log_mutex;

// Constructor: Called by LOG_XXX macros
Logger::Logger(LogLevel message_level, const char *file, int line,
               const char *function_name)
    : m_message_level_instance(message_level),
      m_prefix_and_timestamp_generated(false), m_endl_called(false) {

  // Ensure static members are initialized (idempotent).
  // This is a fallback if Logger::init() wasn't called explicitly at the start
  // of main.
  if (!s_is_initialized_flag) {
    init(); // Initialize with default path and level
  }

  m_should_log_this_message = s_is_initialized_flag && // Log file must be open
                              (static_cast<int>(m_message_level_instance) <=
                               static_cast<int>(s_global_log_level));

  if (m_should_log_this_message) {
    generatePrefix(file, line, function_name);
  }
}

// Destructor: Flushes the buffered message if it's meant to be logged and
// wasn't flushed by std::endl
Logger::~Logger() {
  if (m_should_log_this_message && !m_endl_called && !m_buffer.str().empty()) {
    // Add a newline if std::endl wasn't used and buffer has content
    writeStringToFile(m_buffer.str() + "\n");
  }
}

// Static method to initialize/re-initialize the logger
void Logger::init(const std::string &initial_log_path, LogLevel initial_level) {
  std::lock_guard<std::mutex> lock(s_log_mutex);

  // If already initialized with the same parameters, no need to re-open.
  if (s_is_initialized_flag && initial_log_path == s_log_file_path &&
      initial_level == s_global_log_level) {
    return;
  }

  if (s_log_file_stream.is_open()) {
    s_log_file_stream.close();
  }

  s_log_file_path = initial_log_path;
  s_global_log_level = initial_level;

  try {
    std::filesystem::path logPathFs(s_log_file_path);
    if (logPathFs.has_parent_path()) {
      if (!std::filesystem::exists(logPathFs.parent_path())) {
        if (!std::filesystem::create_directories(logPathFs.parent_path())) {
          // Fallback to console if directory creation fails
          CONSOLE_ERR
              << "LOGGER WARNING: Could not create directory for log file: "
              << logPathFs.parent_path() << ". Log output might fail."
              << std::endl;
        }
      }
    }
  } catch (const std::filesystem::filesystem_error &e) {
    CONSOLE_ERR
        << "LOGGER FS ERROR: Could not create/check directory for log file: "
        << s_log_file_path << " - " << e.what() << std::endl;
  }

  s_log_file_stream.open(s_log_file_path, std::ios::app); // Append mode
  if (!s_log_file_stream.is_open()) {
    CONSOLE_ERR << "LOGGER FATAL ERROR: Could not open log file: "
                << s_log_file_path << ". File logging will be disabled."
                << std::endl;
    s_is_initialized_flag = false;
  } else {
    s_is_initialized_flag = true;
    // Initial log message (this will create a new Logger instance)
    CONSOLE_OUT << "Log file initialized: " << s_log_file_path
                << ". Global log level set to: "
                << static_cast<int>(s_global_log_level) << std::endl;
  }
}

// Static methods to control global logger behavior
void Logger::setGlobalLogLevel(LogLevel level) {
  std::lock_guard<std::mutex> lock(s_log_mutex);
  LogLevel old_level = s_global_log_level;
  s_global_log_level = level;
  if (s_is_initialized_flag &&
      old_level !=
          level) { // Log only if initialized and level actually changed
    Logger(LogLevel::INFO, __FILE__, __LINE__, __PRETTY_FUNCTION__)
        << "Global log level changed from " << static_cast<int>(old_level)
        << " to: " << static_cast<int>(level) << std::endl;
  } else if (!s_is_initialized_flag) {
    // If called before init, init will pick up this new level if we set its
    // default. Or, ensure init is called first in main. For now, init sets its
    // own default. Let's assume init() is called in main() before this
    // typically.
  }
}

LogLevel Logger::getGlobalLogLevel() { return s_global_log_level; }

void Logger::setLogFile(const std::string &file_path) {
  // Re-initialize with the new path, using the current global log level
  init(file_path, s_global_log_level);
}

// Static helper to write the complete, formatted string to file (synchronized)
void Logger::writeStringToFile(const std::string &message) {
  std::lock_guard<std::mutex> lock(s_log_mutex);
  if (s_is_initialized_flag && s_log_file_stream.is_open()) {
    s_log_file_stream << message;
    s_log_file_stream.flush();
  }
  // If not initialized or file not open, message is lost (error already printed
  // during init/open)
}

// Helper to generate prefix (called by constructor if should_log)
void Logger::generatePrefix(const char *file, int line,
                            const char *function_name) {
  // This check is technically redundant if called only when
  // m_should_log_this_message is true, but kept for safety if generatePrefix
  // were ever called directly.
  if (!m_prefix_and_timestamp_generated) {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;

    std::tm tmbuf{};
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&tmbuf, &in_time_t);
#elif defined(__unix__) || defined(__APPLE__) ||                               \
    defined(__linux__) // More comprehensive POSIX check
    localtime_r(&in_time_t, &tmbuf);
#else
    std::tm *temp_tm =
        std::localtime(&in_time_t); // Fallback, potentially not thread-safe
    if (temp_tm)
      tmbuf = *temp_tm;
#endif

    m_buffer << "{";
    const char *base_filename = strrchr(file, '/');
    if (!base_filename)
      base_filename = strrchr(file, '\\');                  // For Windows paths
    m_buffer << (base_filename ? base_filename + 1 : file); // Get only filename

    m_buffer << ":" << line << " (" << function_name << ") ";
    m_buffer << std::put_time(&tmbuf, "%Y-%m-%d %H:%M:%S");
    m_buffer << "." << std::setfill('0') << std::setw(3) << ms.count();
    m_buffer << "} ";
    m_prefix_and_timestamp_generated = true;
  }
}

// Handle std::endl and other ostream manipulators
Logger &Logger::operator<<(std::ostream &(*manip)(std::ostream &)) {
  if (m_should_log_this_message) {
    if (manip == static_cast<std::ostream &(*)(std::ostream &)>(std::endl)) {
      m_buffer << "\n";
      writeStringToFile(m_buffer.str());
      m_buffer.str("");
      m_buffer.clear();
      m_prefix_and_timestamp_generated =
          false; // Ready for next potential part if instance isn't temp
      m_endl_called = true; // Mark that endl was used for this instance
      m_should_log_this_message =
          false; // Effectively "closes" this log operation for this instance
    } else if (manip ==
               static_cast<std::ostream &(*)(std::ostream &)>(std::flush)) {
      writeStringToFile(m_buffer.str()); // Write current buffer content
      // Note: Unlike std::endl, std::flush does not add a newline itself to the
      // buffer s_log_file_stream is flushed by writeStringToFile.
    } else {
      // Apply other manipulators (e.g., std::hex, std::setw, std::fixed) to the
      // internal buffer
      m_buffer << manip;
    }
  }
  return *this;
}