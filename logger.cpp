#include "logger.h" // Or "common.h" if integrated there
#include <iostream> // For CONSOLE_ERR during init failure

// Static member definitions
std::ofstream Logger::s_log_file_stream;
LogLevel Logger::s_global_log_level =
    LogLevel::INFO; // Default global log level
std::string Logger::s_log_file_path = "share/log.txt"; // Default log file path
bool Logger::s_is_initialized_flag = false;
std::mutex Logger::s_log_mutex;

// Constructor: Called by LOG_XXX macros
Logger::Logger(LogLevel message_level, const char *file, int line,
               const char *function_name)
    : m_message_level_instance(message_level),
      m_prefix_and_timestamp_generated(false) {

  // Ensure static members are initialized (idempotent)
  if (!s_is_initialized_flag) {
    init(); // Uses static s_log_file_path and s_global_log_level
  }

  m_should_log_this_message = s_is_initialized_flag && // Log file must be open
                              (static_cast<int>(m_message_level_instance) <=
                               static_cast<int>(s_global_log_level));

  if (m_should_log_this_message) {
    generatePrefix(file, line, function_name);
  }
}

// Destructor: Flushes the buffered message if it's meant to be logged and
// hasn't been flushed by std::endl
Logger::~Logger() {
  if (m_should_log_this_message && !m_buffer.str().empty()) {
    // Add a newline if the user didn't end with std::endl or a manipulator that
    // flushes This handles cases like: LOG_INFO << "message"; (semicolon ends
    // statement)
    writeStringToFile(m_buffer.str() + "\n");
  }
}

// Static method to initialize/re-initialize the logger (e.g., open log file)
void Logger::init(const std::string &initial_log_path, LogLevel initial_level) {
  std::lock_guard<std::mutex> lock(
      s_log_mutex); // Protect static initializations

  if (s_is_initialized_flag && initial_log_path == s_log_file_path &&
      initial_level == s_global_log_level) {
    return; // Already initialized with same parameters
  }

  if (s_log_file_stream.is_open()) {
    s_log_file_stream.close();
  }

  s_log_file_path = initial_log_path;
  s_global_log_level = initial_level; // Set global log level from init too

  // Attempt to create 'share' directory if it doesn't exist.
  // This requires C++17. For older standards, this part might need OS-specific
  // calls or manual creation.
  try {
    std::filesystem::path logPathFs(s_log_file_path);
    if (logPathFs.has_parent_path()) {
      if (!std::filesystem::exists(logPathFs.parent_path())) {
        std::filesystem::create_directories(logPathFs.parent_path());
        // Initial log message below won't capture this if it's the very first
        // one.
      }
    }
  } catch (const std::filesystem::filesystem_error &e) {
    // Use CONSOLE_ERR because logger might not be usable yet
    CONSOLE_ERR << "LOGGER FS ERROR: Could not create directory for log file: "
                << s_log_file_path << " - " << e.what()
                << ". Please ensure 'share' directory exists or can be created."
                << std::endl;
    // Proceed to attempt opening the file anyway; it might fail.
  }

  s_log_file_stream.open(s_log_file_path, std::ios::app); // Append mode
  if (!s_log_file_stream.is_open()) {
    CONSOLE_ERR << "LOGGER FATAL ERROR: Could not open log file: "
                << s_log_file_path << ". File logging will be disabled."
                << std::endl;
    s_is_initialized_flag = false; // Mark as not successfully initialized
  } else {
    s_is_initialized_flag = true;
    // Log successful initialization (this will create a new Logger instance)
    // Note: This relies on the recursive call to the Logger constructor now
    // checking s_is_initialized_flag as true.
    Logger(LogLevel::INFO, __FILE__, __LINE__, __PRETTY_FUNCTION__)
        << "Log file initialized: " << s_log_file_path
        << ". Global log level set to: " << static_cast<int>(s_global_log_level)
        << std::endl;
  }
}

// Static methods to control global logger behavior
void Logger::setGlobalLogLevel(LogLevel level) {
  std::lock_guard<std::mutex> lock(s_log_mutex);
  s_global_log_level = level;
  // Optionally log the change if logger is initialized
  if (s_is_initialized_flag) {
    Logger(LogLevel::INFO, __FILE__, __LINE__, __PRETTY_FUNCTION__)
        << "Global log level changed to: " << static_cast<int>(level)
        << std::endl;
  }
}

LogLevel Logger::getGlobalLogLevel() {
  // No lock needed for reading a LogLevel, assuming it's atomic enough or
  // eventual consistency is fine. For strictness, a lock could be added if
  // reads and writes are frequent and concurrent.
  return s_global_log_level;
}

void Logger::setLogFile(const std::string &file_path) {
  // This will re-initialize with the new path and current global log level.
  init(file_path, s_global_log_level);
}

// Static helper to write the complete, formatted string to file (synchronized)
void Logger::writeStringToFile(const std::string &message) {
  std::lock_guard<std::mutex> lock(s_log_mutex);
  if (s_is_initialized_flag && s_log_file_stream.is_open()) {
    s_log_file_stream << message;
    s_log_file_stream.flush(); // Ensure it's written immediately
  }
}

// Helper to generate prefix (called by constructor if should_log)
void Logger::generatePrefix(const char *file, int line,
                            const char *function_name) {
  if (!m_prefix_and_timestamp_generated) { // Ensure prefix is added only once
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;

    std::tm tmbuf{};
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&tmbuf, &in_time_t);
#elif defined(__unix__) || defined(__APPLE__) || defined(__linux__)
    localtime_r(&in_time_t, &tmbuf);
#else
    // Fallback for other systems (potentially not thread-safe)
    std::tm *temp_tm = std::localtime(&in_time_t);
    if (temp_tm)
      tmbuf = *temp_tm;
#endif

    m_buffer << "{";
    const char *base_filename = strrchr(file, '/'); // POSIX
    if (!base_filename)
      base_filename = strrchr(file, '\\'); // Windows
    m_buffer << (base_filename ? base_filename + 1 : file);

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
      m_buffer << "\n"; // Add newline to buffer
      writeStringToFile(m_buffer.str());
      m_buffer.str(""); // Clear internal buffer
      m_buffer.clear(); // Clear any error flags on the ostringstream
      m_prefix_and_timestamp_generated =
          false; // Reset for a potential (though unlikely with macros) next use
                 // of this specific instance
      m_should_log_this_message =
          false; // Mark this instance as "flushed" for this line
    } else if (manip ==
               static_cast<std::ostream &(*)(std::ostream &)>(std::flush)) {
      // Write whatever is in buffer but don't add newline or clear it unless
      // endl does
      writeStringToFile(m_buffer.str());
      // s_log_file_stream is flushed by writeStringToFile. The internal
      // m_buffer is not cleared by std::flush alone.
    } else {
      m_buffer << manip; // Apply other manipulators (e.g., std::hex) to the
                         // internal buffer
    }
  }
  return *this;
}