#include "common.h" // Includes logger.h
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
// #include <iostream> // Replaced by logger.h
#include <chrono> // For std::chrono::milliseconds
#include <linux/videodev2.h>
#include <map>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread> // For std::this_thread::sleep_for
#include <unistd.h>
#include <vector>

std::map<uint32_t, std::string> capability_descriptions_map = {
    // Renamed to avoid conflict if common.h has one
    {V4L2_CAP_VIDEO_CAPTURE, "Video Capture"},
    {V4L2_CAP_STREAMING, "Streaming"},
    {V4L2_CAP_READWRITE, "Read/Write"},
    {V4L2_CAP_VIDEO_OUTPUT, "Video Output"},
    {V4L2_CAP_VIDEO_OVERLAY, "Video Overlay"},
    {V4L2_CAP_VBI_CAPTURE, "VBI Capture"},
    {V4L2_CAP_TUNER, "Tuner"},
    {V4L2_CAP_AUDIO, "Audio"},
    {V4L2_CAP_RADIO, "Radio"}
    // Add more as needed
};

std::map<uint32_t, std::string> format_descriptions_map = {
    // Renamed
    {V4L2_PIX_FMT_YUYV, "YUYV"},
    {V4L2_PIX_FMT_MJPEG, "MJPEG"},
    {V4L2_PIX_FMT_H264, "H264"},
    {V4L2_PIX_FMT_RGB24, "RGB24"},
    {V4L2_PIX_FMT_BGR24, "BGR24"},
    {V4L2_PIX_FMT_UYVY, "UYVY"},
    {V4L2_PIX_FMT_NV12, "NV12"},
    {V4L2_PIX_FMT_YUV420, "YUV420"}
    // Add more as needed
};

bool captureFrameV4L2(const std::string &device_path, cv::Mat &frame);
bool captureFrameOpenCV(const std::string &device_path, cv::Mat &frame);

std::string getCapabilityDescription(uint32_t cap) {
  std::string description;
  for (const auto &pair : capability_descriptions_map) {
    if (cap & pair.first) {
      if (!description.empty())
        description += ", ";
      description += pair.second;
    }
  }
  return description.empty() ? "Unknown/Other" : description;
}

std::string getFormatDescription(uint32_t format) {
  auto it = format_descriptions_map.find(format);
  if (it != format_descriptions_map.end()) {
    return it->second;
  } else {
    std::stringstream ss;
    char c1 = (format >> 0) & 0xFF;
    char c2 = (format >> 8) & 0xFF;
    char c3 = (format >> 16) & 0xFF;
    char c4 = (format >> 24) & 0xFF;
    ss << "Unknown (";
    if (isprint(c1))
      ss << c1;
    else
      ss << "?";
    if (isprint(c2))
      ss << c2;
    else
      ss << "?";
    if (isprint(c3))
      ss << c3;
    else
      ss << "?";
    if (isprint(c4))
      ss << c4;
    else
      ss << "?";
    ss << " / 0x" << std::hex << std::setw(8) << std::setfill('0') << format
       << std::dec << ")";
    return ss.str();
  }
}

VideoDeviceInfo probeSingleDevice(const std::string &device_path) {
  int fd = -1;
  VideoDeviceInfo deviceInfo = {};
  deviceInfo.device_path = device_path;

  try {
    fd = open(device_path.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) {
      LOG_DEBUG << "Failed to open device " << device_path << " ("
                << strerror(errno) << ")" << std::endl;
      return deviceInfo;
    }

    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
      LOG_WARN << "VIDIOC_QUERYCAP failed for " << device_path << " ("
               << strerror(errno) << ")" << std::endl;
      close(fd);
      return deviceInfo;
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
      LOG_DEBUG << "Device " << device_path
                << " does not support video capture." << std::endl;
      close(fd);
      return deviceInfo;
    }

    deviceInfo.driver_name = reinterpret_cast<char *>(cap.driver);
    deviceInfo.card_name = reinterpret_cast<char *>(cap.card);
    deviceInfo.capabilities = cap.capabilities;

    LOG_DEBUG << "Device " << device_path
              << " opened. Driver: " << deviceInfo.driver_name
              << ", Card: " << deviceInfo.card_name << ", Capabilities: "
              << getCapabilityDescription(cap.capabilities) << " (0x"
              << std::hex << cap.capabilities << std::dec << ")" << std::endl;

    struct v4l2_fmtdesc fmtdesc;
    memset(&fmtdesc, 0, sizeof(fmtdesc));
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    while (ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc) == 0) {
      std::string format_name = getFormatDescription(fmtdesc.pixelformat);
      std::string current_format_details = format_name;

      struct v4l2_frmsizeenum frmsize;
      memset(&frmsize, 0, sizeof(frmsize));
      frmsize.pixel_format = fmtdesc.pixelformat;
      frmsize.index = 0;

      std::string sizes_string = " (Sizes: ";
      bool first_size = true;
      while (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frmsize) == 0) {
        if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
          if (!first_size)
            sizes_string += ", ";
          sizes_string += std::to_string(frmsize.discrete.width) + "x" +
                          std::to_string(frmsize.discrete.height);
          first_size = false;
        } else if (frmsize.type == V4L2_FRMSIZE_TYPE_STEPWISE) {
          if (!first_size)
            sizes_string += ", ";
          sizes_string +=
              "Stepwise (Min: " + std::to_string(frmsize.stepwise.min_width) +
              "x" + std::to_string(frmsize.stepwise.min_height) +
              " Max: " + std::to_string(frmsize.stepwise.max_width) + "x" +
              std::to_string(frmsize.stepwise.max_height) +
              " Step: " + std::to_string(frmsize.stepwise.step_width) + "x" +
              std::to_string(frmsize.stepwise.step_height) + ")";
          first_size = false;
        }
        frmsize.index++;
      }
      if (first_size)
        sizes_string += "N/A or Default";
      sizes_string += ")";
      current_format_details += sizes_string;

      deviceInfo.supported_format_details.push_back(current_format_details);
      LOG_DEBUG << "  Found format " << fmtdesc.index << ": " << format_name
                << " (0x" << std::hex << fmtdesc.pixelformat << std::dec
                << ") - Details: " << current_format_details << std::endl;
      fmtdesc.index++;
    }
  } catch (const std::runtime_error &e) {
    LOG_ERROR << "Runtime error probing " << device_path << ": " << e.what()
              << std::endl;
  }
  if (fd != -1) {
    close(fd);
  }
  return deviceInfo;
}

std::vector<VideoDeviceInfo> probeVideoDevices(int max_devices) {
  std::vector<VideoDeviceInfo> devices;
  for (int i = 0; i < max_devices; ++i) {
    std::string device_path = "/dev/video" + std::to_string(i);
    struct stat buffer;
    if (stat(device_path.c_str(), &buffer) == 0) {
      VideoDeviceInfo deviceInfo = probeSingleDevice(device_path);
      if (!deviceInfo.driver_name.empty() ||
          !deviceInfo.card_name.empty()) { // If any info was retrieved
        devices.push_back(deviceInfo);
      }
      LOG_DEBUG << "Device " << device_path << " exists." << std::endl;
    } else {
      LOG_DEBUG << "Device " << device_path << " does not exist." << std::endl;
    }
  }
  return devices;
}

bool captureSnapshot(const std::string &device_path,
                     const std::string &output_path) {
  cv::Mat frame;
  bool success = false;
  LOG_INFO << "Attempting snapshot from " << device_path << " using mode: "
           << (gCaptureMode == MODE_OPENCV ? "OpenCV" : "V4L2") << std::endl;
  try {
    if (gCaptureMode == MODE_OPENCV) {
      success = captureFrameOpenCV(device_path, frame);
    } else {
      success = captureFrameV4L2(device_path, frame);
    }

    if (success) {
      if (!frame.empty()) {
        if (!cv::imwrite(output_path, frame)) {
          LOG_ERROR << "Failed to save image: " << output_path << std::endl;
          THROWGEMERROR("Failed to save image: " + output_path);
        }
        LOG_INFO << "Snapshot saved to " << output_path << std::endl;
      } else {
        LOG_WARN << "captureFrame returned true but frame is empty for device "
                 << device_path << "." << std::endl;
        return false;
      }
    } else {
      LOG_ERROR << "captureFrame failed for device " << device_path
                << " using mode "
                << (gCaptureMode == MODE_OPENCV ? "OpenCV" : "V4L2")
                << std::endl;
      return false;
    }
  } catch (const GEMError &e) {
    LOG_ERROR << "GEMError during snapshot from " << device_path << ": "
              << e.what() << std::endl;
    return false;
  } catch (const std::runtime_error &e) {
    LOG_ERROR << "Runtime error during snapshot from " << device_path << ": "
              << e.what() << std::endl;
    return false;
  }
  return true;
}

bool captureFrameV4L2(const std::string &device_path, cv::Mat &frame) {
  LOG_DEBUG << "Attempting V4L2 capture from " << device_path
            << " for resolution " << g_capture_width << "x" << g_capture_height
            << std::endl;
  int fd = -1;
  void *buffer_mmap = MAP_FAILED;         // Initialize to MAP_FAILED
  struct v4l2_buffer buf_mmap_info = {0}; // To store length for munmap

  try {
    fd = open(device_path.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) {
      LOG_ERROR << "Failed to open device " << device_path
                << " for V4L2 capture: " << strerror(errno) << std::endl;
      THROWGEMERROR("Failed to open device " + device_path +
                    " for V4L2 capture");
    }
    LOG_DEBUG << "Device " << device_path
              << " opened for V4L2 capture. FD: " << fd << std::endl;

    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
      LOG_ERROR << "VIDIOC_QUERYCAP failed for " << device_path << ": "
                << strerror(errno) << std::endl;
      THROWGEMERROR("VIDIOC_QUERYCAP failed for " + device_path);
    }
    LOG_DEBUG << "VIDIOC_QUERYCAP successful for " << device_path << std::endl;

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
      LOG_ERROR << "Device " << device_path
                << " does not support video capture." << std::endl;
      THROWGEMERROR("Device " + device_path +
                    " does not support video capture");
    }
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
      LOG_ERROR << "Device " << device_path << " does not support streaming."
                << std::endl;
      THROWGEMERROR("Device " + device_path + " does not support streaming");
    }
    LOG_DEBUG << "Device " << device_path
              << " supports video capture and streaming." << std::endl;

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = g_capture_width;
    fmt.fmt.pix.height = g_capture_height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_NONE; // Changed from INTERLACED to NONE/ANY
                                         // for progressive common on webcams

    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
      LOG_WARN << "Failed to set MJPEG format at " << g_capture_width << "x"
               << g_capture_height << " for " << device_path
               << ". Error: " << strerror(errno) << ". Trying YUYV."
               << std::endl;
      fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
      if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        LOG_ERROR << "Failed to set YUYV format at " << g_capture_width << "x"
                  << g_capture_height << " for " << device_path
                  << ". Error: " << strerror(errno) << std::endl;
        THROWGEMERROR(std::string("Failed to set YUYV format at ") +
                      Num2Str(g_capture_width).str() + "x" +
                      Num2Str(g_capture_height).str() + " for " + device_path);
      }
    }
    LOG_DEBUG << "VIDIOC_S_FMT attempted for " << device_path << " "
              << g_capture_width << "x" << g_capture_height
              << ". Actual format set: Width=" << fmt.fmt.pix.width
              << ", Height=" << fmt.fmt.pix.height << ", PixelFormat="
              << getFormatDescription(fmt.fmt.pix.pixelformat) << " (0x"
              << std::hex << fmt.fmt.pix.pixelformat << std::dec << ")"
              << std::endl;

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
      LOG_ERROR << "VIDIOC_REQBUFS failed for " << device_path << ": "
                << strerror(errno) << std::endl;
      THROWGEMERROR("VIDIOC_REQBUFS failed for " + device_path);
    }
    LOG_DEBUG << "VIDIOC_REQBUFS successful. Device " << device_path
              << " allocated " << req.count << " buffers." << std::endl;
    if (req.count < 1) {
      LOG_ERROR << "Insufficient buffer memory on " << device_path << "."
                << std::endl;
      THROWGEMERROR("Insufficient buffer memory on " + device_path);
    }

    // Store buffer info for mmap and munmap
    buf_mmap_info.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf_mmap_info.memory = V4L2_MEMORY_MMAP;
    buf_mmap_info.index = 0;
    if (ioctl(fd, VIDIOC_QUERYBUF, &buf_mmap_info) < 0) {
      LOG_ERROR << "VIDIOC_QUERYBUF failed for " << device_path << ": "
                << strerror(errno) << std::endl;
      THROWGEMERROR("VIDIOC_QUERYBUF failed for " + device_path);
    }
    LOG_DEBUG << "VIDIOC_QUERYBUF successful for " << device_path
              << ". Buffer length: " << buf_mmap_info.length
              << ", Offset: " << buf_mmap_info.m.offset << std::endl;

    buffer_mmap = mmap(NULL, buf_mmap_info.length, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, buf_mmap_info.m.offset);
    if (buffer_mmap == MAP_FAILED) {
      LOG_ERROR << "mmap failed for " << device_path << ": " << strerror(errno)
                << std::endl;
      THROWGEMERROR("mmap failed for " + device_path);
    }
    LOG_DEBUG << "mmap successful for " << device_path
              << ". Buffer address: " << buffer_mmap << std::endl;

    if (ioctl(fd, VIDIOC_QBUF, &buf_mmap_info) < 0) {
      LOG_ERROR << "VIDIOC_QBUF failed for " << device_path << ": "
                << strerror(errno) << std::endl;
      THROWGEMERROR("VIDIOC_QBUF failed for " + device_path);
    }
    LOG_DEBUG << "VIDIOC_QBUF successful for " << device_path << std::endl;

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
      LOG_ERROR << "VIDIOC_STREAMON failed for " << device_path << ": "
                << strerror(errno) << std::endl;
      THROWGEMERROR("VIDIOC_STREAMON failed for " + device_path);
    }
    LOG_DEBUG << "VIDIOC_STREAMON successful for " << device_path << std::endl;

    fd_set fds;
    struct timeval tv;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    tv.tv_sec = 2;
    tv.tv_usec = 0;
    int r = select(fd + 1, &fds, NULL, NULL, &tv);
    if (r < 0) {
      LOG_ERROR << "select failed for " << device_path << ": "
                << strerror(errno) << std::endl;
      THROWGEMERROR("select failed for " + device_path);
    }
    if (r == 0) {
      LOG_ERROR << "Timeout waiting for frame from " << device_path
                << std::endl;
      THROWGEMERROR("Timeout waiting for frame from " + device_path);
    }
    LOG_DEBUG << "select successful. Frame ready on " << device_path
              << std::endl;

    if (ioctl(fd, VIDIOC_DQBUF, &buf_mmap_info) < 0) {
      LOG_ERROR << "VIDIOC_DQBUF failed for " << device_path << ": "
                << strerror(errno) << std::endl;
      THROWGEMERROR("VIDIOC_DQBUF failed for " + device_path);
    }
    LOG_DEBUG << "VIDIOC_DQBUF successful for " << device_path
              << ". Bytes used: " << buf_mmap_info.bytesused << std::endl;

    if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_MJPEG) {
      std::vector<uchar> data_vec(static_cast<unsigned char *>(buffer_mmap),
                                  static_cast<unsigned char *>(buffer_mmap) +
                                      buf_mmap_info.bytesused);
      frame = cv::imdecode(cv::Mat(data_vec), cv::IMREAD_COLOR);
      if (frame.empty()) {
        LOG_ERROR << "Error decoding MJPEG frame from " << device_path
                  << std::endl;
        THROWGEMERROR("Error decoding MJPEG frame from " + device_path);
      }
      LOG_DEBUG << "MJPEG frame decoded from " << device_path << std::endl;
    } else if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_YUYV) {
      cv::Mat yuyv_frame(cv::Size(fmt.fmt.pix.width, fmt.fmt.pix.height),
                         CV_8UC2, buffer_mmap);
      cv::cvtColor(yuyv_frame, frame, cv::COLOR_YUV2BGR_YUYV);
      LOG_DEBUG << "YUYV frame converted to BGR from " << device_path
                << std::endl;
    } else {
      LOG_ERROR << "Unsupported pixel format 0x" << std::hex
                << fmt.fmt.pix.pixelformat << std::dec << " from "
                << device_path << " for direct saving." << std::endl;
      THROWGEMERROR("Unsupported V4L2 pixel format for processing: " +
                    getFormatDescription(fmt.fmt.pix.pixelformat));
    }

    if (frame.empty()) {
      LOG_ERROR << "No frame data captured from " << device_path << "."
                << std::endl;
      THROWGEMERROR("No frame data captured from " + device_path);
    }
    LOG_DEBUG << "Frame data captured from " << device_path << std::endl;

    if (ioctl(fd, VIDIOC_STREAMOFF, &type) < 0) {
      LOG_WARN << "VIDIOC_STREAMOFF failed for " << device_path << ": "
               << strerror(errno)
               << std::endl; /* Not throwing, attempt cleanup */
    } else {
      LOG_DEBUG << "VIDIOC_STREAMOFF successful for " << device_path
                << std::endl;
    }

  } catch (const GEMError &e) { // Catch specific GEMError to log and rethrow
    LOG_ERROR << "GEMError during V4L2 capture from " << device_path << ": "
              << e.what() << std::endl;
    if (buffer_mmap != MAP_FAILED)
      munmap(buffer_mmap, buf_mmap_info.length);
    if (fd != -1)
      close(fd);
    throw; // Re-throw to be caught by caller if necessary
  } catch (const std::runtime_error &e) { // Catch other runtime errors
    LOG_ERROR << "Runtime error during V4L2 capture from " << device_path
              << ": " << e.what() << std::endl;
    if (buffer_mmap != MAP_FAILED)
      munmap(buffer_mmap, buf_mmap_info.length);
    if (fd != -1)
      close(fd);
    return false; // Indicate failure
  }

  // Cleanup outside catch if no exception path was taken
  if (buffer_mmap != MAP_FAILED) {
    if (munmap(buffer_mmap, buf_mmap_info.length) < 0) {
      LOG_WARN << "munmap failed for " << device_path
               << " during cleanup: " << strerror(errno) << std::endl;
    } else {
      LOG_DEBUG << "munmap successful for " << device_path << " during cleanup."
                << std::endl;
    }
  }
  if (fd != -1) {
    close(fd);
  }
  return !frame.empty();
}

bool captureFrameOpenCV(const std::string &device_path, cv::Mat &frame) {
  int camera_index = 0;
  size_t last_digit_pos = device_path.find_last_of("0123456789");
  if (last_digit_pos != std::string::npos) {
    size_t first_digit_pos = last_digit_pos;
    while (first_digit_pos > 0 && isdigit(device_path[first_digit_pos - 1])) {
      first_digit_pos--;
    }
    try {
      camera_index = std::stoi(device_path.substr(first_digit_pos));
    } catch (const std::exception &e) { // More specific catch
      LOG_WARN << "Could not parse camera index from device path '"
               << device_path << "'. Using default index 0. Error: " << e.what()
               << std::endl;
      camera_index = 0;
    }
  } else {
    LOG_DEBUG
        << "No numeric index found in device path '" << device_path
        << "'. Assuming index 0 for OpenCV if it's a path like '/dev/video0'."
        << std::endl;
    // OpenCV can often take /dev/videoX directly, or an integer.
    // If it's not a numeric string, VideoCapture might try to open it as a
    // file/url. For /dev/videoX, just using the number is safer.
  }

  LOG_DEBUG << "Attempting OpenCV capture from index " << camera_index
            << " (derived from " << device_path << ") for resolution "
            << g_capture_width << "x" << g_capture_height << std::endl;

  cv::VideoCapture cap;
  cap.open(camera_index, cv::CAP_V4L2);

  if (!cap.isOpened()) {
    LOG_WARN << "Opening camera with CAP_V4L2 failed for index " << camera_index
             << ", trying default backend." << std::endl;
    cap.open(camera_index); // Fallback to default backend
    if (!cap.isOpened()) {
      LOG_ERROR << "Cannot open video capture device with index "
                << camera_index << " using OpenCV." << std::endl;
      THROWGEMERROR(
          std::string("Cannot open video capture device with index ") +
          Num2Str(camera_index).str() + " using OpenCV.");
    }
  }
  LOG_DEBUG << "OpenCV VideoCapture opened for index " << camera_index << "."
            << std::endl;

  if (!trySetCameraResolution(cap, g_capture_width, g_capture_height, true)) {
    int final_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int final_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (final_width != g_capture_width || final_height != g_capture_height) {
      std::string msg = "OpenCV Capture: Failed to set desired resolution " +
                        Num2Str(g_capture_width).str() + "x" +
                        Num2Str(g_capture_height).str() +
                        ". Actual resolution is " + Num2Str(final_width).str() +
                        "x" + Num2Str(final_height).str() + ".";
      LOG_ERROR << msg << std::endl;
      cap.release();
      THROWGEMERROR(msg);
    }
  }

  LOG_DEBUG << "Attempting to grab and read frame using OpenCV..." << std::endl;
  cap.grab(); // Grab a few frames
  std::this_thread::sleep_for(
      std::chrono::milliseconds(50)); // Small delay for camera
  bool success = cap.read(frame);

  if (!success || frame.empty()) {
    LOG_ERROR << "Failed to read frame from device index " << camera_index
              << " using OpenCV." << std::endl;
    cap.release();
    return false;
  }

  LOG_INFO
      << "Successfully captured frame using OpenCV VideoCapture from index "
      << camera_index << "." << std::endl;
  cap.release();
  return true;
}

bool captureFrame(const std::string &device_path, cv::Mat &captured_image) {
  if (gCaptureMode == MODE_OPENCV) {
    return captureFrameOpenCV(device_path, captured_image);
  } else {
    return captureFrameV4L2(device_path, captured_image);
  }
}