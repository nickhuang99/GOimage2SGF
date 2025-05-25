#include "common.h"
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <linux/videodev2.h>
#include <map>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream> // For stringstream
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

// Simplified Capability code to human-readable text mapping for webcams
std::map<uint32_t, std::string> capability_descriptions = {
    {V4L2_CAP_VIDEO_CAPTURE, "Video Capture"},
    {V4L2_CAP_STREAMING, "Streaming"},
    {V4L2_CAP_READWRITE, "Read/Write"},
};

// Simplified Pixel format code to human-readable text mapping for webcams
std::map<uint32_t, std::string> format_descriptions = {
    {V4L2_PIX_FMT_YUYV, "YUYV"},   {V4L2_PIX_FMT_MJPEG, "MJPEG"},
    {V4L2_PIX_FMT_H264, "H264"},   {V4L2_PIX_FMT_RGB24, "RGB24"},
    {V4L2_PIX_FMT_BGR24, "BGR24"}, {V4L2_PIX_FMT_UYVY, "UYVY"},
    {V4L2_PIX_FMT_NV12, "NV12"},
};

// Rename original V4L2 capture function
bool captureFrameV4L2(const std::string &device_path, cv::Mat &frame);
// Add declaration for new OpenCV capture function
bool captureFrameOpenCV(const std::string &device_path, cv::Mat &frame);

// Function to get human-readable description for a capability
std::string getCapabilityDescription(uint32_t cap) {
  std::string description;
  for (const auto &pair : capability_descriptions) {
    if (cap & pair.first) {
      if (!description.empty()) {
        description += ", ";
      }
      description += pair.second;
    }
  }
  return description.empty() ? "Unknown Capabilities" : description;
}

// Function to get human-readable description for a pixel format
std::string getFormatDescription(uint32_t format) {
  auto it = format_descriptions.find(format);
  if (it != format_descriptions.end()) {
    return it->second;
  } else {
    std::stringstream ss;
    ss << "Unknown Format (0x" << std::hex << std::setw(8) << std::setfill('0')
       << format << ")";
    return ss.str();
  }
}

// Function to probe a single video device
VideoDeviceInfo probeSingleDevice(const std::string &device_path) {
  int fd = -1;
  VideoDeviceInfo deviceInfo = {};
  deviceInfo.device_path = device_path;

  try {
    fd = open(device_path.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) {
      if (bDebug) {
        std::cerr << "Debug: Failed to open device " << device_path << " ("
                  << strerror(errno) << ")\n";
      }
      return deviceInfo; // Return empty deviceInfo
    }

    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
      std::cerr << "Warning: VIDIOC_QUERYCAP failed for " << device_path << " ("
                << strerror(errno) << ")\n";
      close(fd);
      return deviceInfo; // Return empty deviceInfo
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
      if (bDebug) {
        std::cerr << "Debug: Device " << device_path
                  << " does not support video capture\n";
      }
      close(fd);
      return deviceInfo; // Return empty deviceInfo
    }

    deviceInfo.driver_name = reinterpret_cast<char *>(cap.driver);
    deviceInfo.card_name = reinterpret_cast<char *>(cap.card);
    deviceInfo.capabilities = cap.capabilities;

    if (bDebug) {
      std::cout << "Debug: Device " << device_path
                << " opened. Driver: " << deviceInfo.driver_name
                << ", Card: " << deviceInfo.card_name << ", Capabilities: "
                << getCapabilityDescription(cap.capabilities) << " (0x"
                << std::hex << cap.capabilities << std::dec << ")\n";
    }

    struct v4l2_fmtdesc fmtdesc;
    memset(&fmtdesc, 0, sizeof(fmtdesc));
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    while (ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc) == 0) {
      std::string format_name = getFormatDescription(fmtdesc.pixelformat);
      std::string current_format_details = format_name;

      // Enumerate frame sizes for this format
      struct v4l2_frmsizeenum frmsize;
      memset(&frmsize, 0, sizeof(frmsize));
      frmsize.pixel_format = fmtdesc.pixelformat;
      frmsize.index = 0;

      std::string sizes_string = " (Sizes: ";
      bool first_size = true;
      while (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frmsize) == 0) {
        if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
          if (!first_size) {
            sizes_string += ", ";
          }
          sizes_string += std::to_string(frmsize.discrete.width) + "x" +
                          std::to_string(frmsize.discrete.height);
          first_size = false;
        } else if (frmsize.type == V4L2_FRMSIZE_TYPE_STEPWISE) {
          if (!first_size) {
            sizes_string += ", ";
          }
          sizes_string +=
              "Stepwise (Min: " + std::to_string(frmsize.stepwise.min_width) +
              "x" + std::to_string(frmsize.stepwise.min_height) +
              " Max: " + std::to_string(frmsize.stepwise.max_width) + "x" +
              std::to_string(frmsize.stepwise.max_height) +
              " Step: " + std::to_string(frmsize.stepwise.step_width) + "x" +
              std::to_string(frmsize.stepwise.step_height) + ")";
          first_size = false;
        }
        // You could also enumerate frame intervals (FPS) here for each size if
        // needed, using VIDIOC_ENUM_FRAMEINTERVALS
        frmsize.index++;
      }
      if (first_size) { // No sizes found or ENUM_FRAMESIZES failed for the
                        // first index
        // This can happen if the format doesn't support size enumeration or
        // only supports a default size not listed this way.
        sizes_string += "N/A or Default";
      }
      sizes_string += ")";
      current_format_details += sizes_string;

      deviceInfo.supported_format_details.push_back(current_format_details);

      if (bDebug) {
        // The detailed print is now part of the collected string,
        // but we can still log the basic format finding.
        std::cout << "  Debug: Found format " << fmtdesc.index << ": "
                  << format_name << " (0x" << std::hex << fmtdesc.pixelformat
                  << std::dec << ") - Details: " << current_format_details
                  << "\n";
      }
      fmtdesc.index++;
    }
  } catch (const std::runtime_error
               &e) { // Should ideally not be hit if we check returns
    std::cerr << "Error probing " << device_path << ": " << e.what()
              << std::endl;
  }
  if (fd != -1) {
    close(fd);
  }
  return deviceInfo;
}

// Function to probe all potential video devices
std::vector<VideoDeviceInfo> probeVideoDevices(int max_devices) {
  std::vector<VideoDeviceInfo> devices;
  for (int i = 0; i < max_devices; ++i) {
    std::string device_path = "/dev/video" + std::to_string(i);
    struct stat buffer;
    if (stat(device_path.c_str(), &buffer) == 0) {
      VideoDeviceInfo deviceInfo = probeSingleDevice(device_path);
      if (!deviceInfo.driver_name.empty()) {
        devices.push_back(deviceInfo);
      }
    }
    if (bDebug) {
      std::cout << "Debug: Device " << device_path
                << (stat(device_path.c_str(), &buffer) == 0
                        ? " exists\n"
                        : " does not exist\n");
    }
  }
  return devices;
}

// --- MODIFIED captureSnapshot to select capture method ---
bool captureSnapshot(const std::string &device_path,
                     const std::string &output_path) {
  cv::Mat frame;
  bool success = false;

  try {
    // Select capture function based on global mode
    if (gCaptureMode == MODE_OPENCV) {
      success = captureFrameOpenCV(device_path, frame);
    } else { // Default or explicitly MODE_V4L2
      success = captureFrameV4L2(device_path, frame);
    }

    if (success) {
      if (!frame.empty()) {
        if (!cv::imwrite(output_path, frame)) {
          THROWGEMERROR("Failed to save image: " + output_path);
        }
        if (bDebug)
          std::cout << "Debug: Snapshot saved to " << output_path << std::endl;
      } else {
        std::cerr
            << "Warning: captureFrame returned true but frame is empty.\n";
        return false;
      }
    } else {
      std::cerr << "Error: captureFrame failed for device " << device_path
                << " using mode "
                << (gCaptureMode == MODE_OPENCV ? "OpenCV" : "V4L2")
                << std::endl;
      return false;
    }
  } catch (
      const std::runtime_error &e) { // Catch GEMError or std::runtime_error
    std::cerr << "Error during snapshot: " << e.what() << std::endl;
    return false;
  }
  return true;
}

bool captureFrameV4L2(const std::string &device_path, cv::Mat &frame) {
  int fd = -1;
  try {
    fd = open(device_path.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) {
      THROWGEMERROR("Failed to open device for capture");
    }
    if (bDebug) {
      std::cout << "Debug: Device " << device_path << " opened for capture.\n";
    }

    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
      THROWGEMERROR("VIDIOC_QUERYCAP during capture");
    }
    if (bDebug) {
      std::cout << "Debug: VIDIOC_QUERYCAP successful.\n";
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
      THROWGEMERROR("Device does not support video capture");
    }
    if (bDebug) {
      std::cout << "Debug: Device supports video capture.\n";
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
      THROWGEMERROR("Device does not support streaming");
    }
    if (bDebug) {
      std::cout << "Debug: Device supports streaming.\n";
    }

    // Set video format
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = g_capture_width;   // USE GLOBAL
    fmt.fmt.pix.height = g_capture_height; // USE GLOBAL
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field =
        V4L2_FIELD_INTERLACED; // Or V4L2_FIELD_NONE if progressive

    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
      std::cerr << "Warning: Failed to set MJPEG format at " << g_capture_width
                << "x" << g_capture_height << ". Trying YUYV." << std::endl;
      // Keep same requested dimensions for YUYV attempt
      fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
      if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        THROWGEMERROR(std::string("Failed to set YUYV format at ") +
                      Num2Str(g_capture_width).str() + "x" +
                      Num2Str(g_capture_height).str());
      }
    }
    if (bDebug) {
      std::cout << "Debug: VIDIOC_S_FMT attempted for " << g_capture_width
                << "x" << g_capture_height
                << ". Actual format set: Width=" << fmt.fmt.pix.width
                << ", Height=" << fmt.fmt.pix.height << ", PixelFormat="
                << getFormatDescription(fmt.fmt.pix.pixelformat) << " (0x"
                << std::hex << fmt.fmt.pix.pixelformat << std::dec << ")\n";
    }
    // Request buffers
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
      THROWGEMERROR("VIDIOC_REQBUFS");
    }
    if (bDebug) {
      std::cout << "Debug: VIDIOC_REQBUFS successful. Requested " << req.count
                << " buffers.\n";
    }

    if (req.count < 1) {
      THROWGEMERROR("Insufficient buffer memory");
    }
    if (bDebug) {
      std::cout << "Debug: At least 1 buffer allocated.\n";
    }

    // Map the buffer to user space
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
      THROWGEMERROR("VIDIOC_QUERYBUF");
    }
    if (bDebug) {
      std::cout << "Debug: VIDIOC_QUERYBUF successful.\n";
    }

    void *buffer = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                        fd, buf.m.offset);
    if (buffer == MAP_FAILED) {
      THROWGEMERROR("mmap");
    }
    if (bDebug) {
      std::cout << "Debug: mmap successful. Buffer address: " << buffer
                << ", Length: " << buf.length << "\n";
    }

    // Queue the buffer
    if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
      THROWGEMERROR("VIDIOC_QBUF");
    }
    if (bDebug) {
      std::cout << "Debug: VIDIOC_QBUF successful.\n";
    }

    // Start capturing
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
      THROWGEMERROR("VIDIOC_STREAMON");
    }
    if (bDebug) {
      std::cout << "Debug: VIDIOC_STREAMON successful.\n";
    }

    // Wait for a frame to be ready
    fd_set fds;
    struct timeval tv;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    tv.tv_sec = 2;
    tv.tv_usec = 0;
    int r = select(fd + 1, &fds, NULL, NULL, &tv);
    if (r < 0) {
      THROWGEMERROR("select");
    }
    if (r == 0) {
      THROWGEMERROR("Timeout waiting for frame");
    }
    if (bDebug) {
      std::cout << "Debug: select successful. A frame is ready.\n";
    }

    // Dequeue the buffer
    if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
      THROWGEMERROR("VIDIOC_DQBUF");
    }
    if (bDebug) {
      std::cout << "Debug: VIDIOC_DQBUF successful.\n";
    }

    // Save the captured frame to a file using OpenCV
    if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_MJPEG) {
      std::vector<uchar> data(static_cast<unsigned char *>(buffer),
                              static_cast<unsigned char *>(buffer) +
                                  buf.bytesused);
      frame = cv::imdecode(cv::Mat(data), cv::IMREAD_COLOR);
      if (frame.empty()) {
        THROWGEMERROR("Error decoding MJPEG frame");
      }
      if (bDebug) {
        std::cout << "Debug: MJPEG frame decoded.\n";
      }
    } else if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_YUYV) {
      cv::Mat yuyv_frame(cv::Size(fmt.fmt.pix.width, fmt.fmt.pix.height),
                         CV_8UC2, buffer);
      cv::cvtColor(yuyv_frame, frame, cv::COLOR_YUV2BGR_YUYV);
      if (bDebug) {
        std::cout << "Debug: YUYV frame converted to BGR.\n";
      }
    } else {
      std::cerr << "Error: Unsupported pixel format 0x" << std::hex
                << fmt.fmt.pix.pixelformat << std::dec
                << " for direct saving.\n";
    }

    if (frame.empty()) {
      THROWGEMERROR("error: No frame data captured.");
    }
    if (bDebug) {
      std::cout << "Debug: Frame data captured.\n";
    }

    // Stop capturing
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) < 0) {
      THROWGEMERROR("VIDIOC_STREAMOFF");
    }
    if (bDebug) {
      std::cout << "Debug: VIDIOC_STREAMOFF successful.\n";
    }

    // Unmap the buffer
    if (munmap(buffer, buf.length) < 0) {
      THROWGEMERROR("munmap");
    }
    if (bDebug) {
      std::cout << "Debug: munmap successful.\n";
    }

    close(fd);
    return !frame.empty();

  } catch (const std::runtime_error &e) {
    std::cerr << "Error during capture: " << e.what() << std::endl;
    if (fd != -1) {
      close(fd);
    }
    return false;
  }
}

// --- NEW captureFrame using OpenCV ---
bool captureFrameOpenCV(const std::string &device_path, cv::Mat &frame) {
  int camera_index = 0; // Default
  // Basic parsing logic (same as before)
  size_t last_digit_pos = device_path.find_last_of("0123456789");
  if (last_digit_pos != std::string::npos) {
    size_t first_digit_pos = last_digit_pos;
    while (first_digit_pos > 0 && isdigit(device_path[first_digit_pos - 1])) {
      first_digit_pos--;
    }
    try {
      camera_index = std::stoi(device_path.substr(first_digit_pos));
    } catch (...) {
      camera_index = 0;
    }
  }
  if (bDebug)
    std::cout << "Debug: Attempting capture from index " << camera_index
              << " (derived from " << device_path << ") using OpenCV."
              << std::endl;

  cv::VideoCapture cap;
  // Use the direct index for VideoCapture
  cap.open(camera_index,
           cv::CAP_V4L2); // Explicitly suggest V4L2 backend for OpenCV on Linux

  if (!cap.isOpened()) {
    THROWGEMERROR(
        std::string("Error: Cannot open video capture device with index ") +
        Num2Str(camera_index).str() + std::string(" using OpenCV."));
  }
  if (bDebug)
    std::cout << "Debug: OpenCV - Requesting frame size " << g_capture_width
              << "x" << g_capture_height << "." << std::endl;
  // Use the new utility function
  if (!trySetCameraResolution(cap, g_capture_width, g_capture_height, true)) {
    // If strict resolution is required for snapshot, you might throw or log an
    // error. For now, we'll proceed with whatever resolution the camera settled
    // on if trySet fails. The utility function logs the failure in debug mode.
    // If you need to THROW here if it doesn't match g_capture_width/height:
    int final_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int final_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (final_width != g_capture_width || final_height != g_capture_height) {
      std::stringstream ss;
      ss << "Snapshot: Failed to set desired resolution " << g_capture_width
         << "x" << g_capture_height << ". Actual resolution is " << final_width
         << "x" << final_height << ".";
      THROWGEMERROR(ss.str()); // Or handle differently for snapshots
    }
  }

  // Optional: Allow camera to warm up slightly
  // std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Requires
  // <thread>, <chrono>
  cap.grab(); // Grab a few frames to allow settings to stabilize

  bool success = cap.read(frame); // Read one frame

  if (!success || frame.empty()) {
    std::cerr << "Error: Failed to read frame from device index "
              << camera_index << " using OpenCV." << std::endl;
    cap.release();
    return false;
  }

  if (bDebug)
    std::cout << "Debug: Successfully captured frame using OpenCV VideoCapture."
              << std::endl;

  cap.release(); // Release camera immediately after capture
  return true;
}

bool captureFrame(const std::string &device_path, cv::Mat &captured_image) {
  // Select capture function based on global mode
  if (gCaptureMode == MODE_OPENCV) {
    return captureFrameOpenCV(device_path, captured_image);
  } else { // Default or explicitly MODE_V4L2
    return captureFrameV4L2(device_path, captured_image);
  }
}