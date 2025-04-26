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

extern bool bDebug;
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

// Error handling helper function
void errno_exit(const char *s) {
  throw std::runtime_error(s + std::string(" error ") + std::to_string(errno));
}

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
      return deviceInfo;
    }

    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
      std::cerr << "Warning: VIDIOC_QUERYCAP failed for " << device_path << " ("
                << strerror(errno) << ")\n";
      close(fd);
      return deviceInfo;
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
      close(fd);
      return deviceInfo;
    }

    deviceInfo.driver_name = reinterpret_cast<char *>(cap.driver);
    deviceInfo.card_name = reinterpret_cast<char *>(cap.card);
    deviceInfo.capabilities = cap.capabilities;

    struct v4l2_fmtdesc fmtdesc;
    memset(&fmtdesc, 0, sizeof(fmtdesc));
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    while (ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc) == 0) {
      if (format_descriptions.find(fmtdesc.pixelformat) !=
          format_descriptions.end()) {
        deviceInfo.supported_formats.push_back(fmtdesc.pixelformat);
      }
      fmtdesc.index++;
    }

  } catch (const std::runtime_error &e) {
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
  }
  return devices;
}

// Function to capture a snapshot (remains mostly the same)
bool captureSnapshot(const std::string &device_path,
                     const std::string &output_path) {
  cv::Mat frame;
  try {
    if (captureFrame(device_path, frame)) {
      if (!frame.empty() && !cv::imwrite(output_path, frame)) {
        throw std::runtime_error("Failed to save image");
      } else if (frame.empty()) {
        std::cerr << "Warning: No frame data to save.\n";
        return false;
      }
    } else return false;

  } catch (const std::runtime_error &e) {
    std::cerr << "Error during capture: " << e.what() << std::endl;
    return false;
  }
  return true;
}

bool captureFrame(const std::string &device_path, cv::Mat &frame) {
  int fd = -1;
  try {
    fd = open(device_path.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) {
      errno_exit("Failed to open device for capture");
    }

    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
      errno_exit("VIDIOC_QUERYCAP during capture");
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
      throw std::runtime_error("Device does not support video capture");
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
      throw std::runtime_error("Device does not support streaming");
    }

    // Set video format to MJPEG
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 640;
    fmt.fmt.pix.height = 480;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
      std::cerr << "Warning: Failed to set MJPEG format, trying YUYV."
                << std::endl;
      fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
      if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        errno_exit("Failed to set YUYV format");
      }
    }
    std::cerr << "Using format: Width=" << fmt.fmt.pix.width
              << ", Height=" << fmt.fmt.pix.height << ", PixelFormat="
              << getFormatDescription(fmt.fmt.pix.pixelformat) << " (0x"
              << std::hex << fmt.fmt.pix.pixelformat << std::dec << ")\n";

    // Request buffers
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
      errno_exit("VIDIOC_REQBUFS");
    }

    if (req.count < 1) {
      throw std::runtime_error("Insufficient buffer memory");
    }

    // Map the buffer to user space
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
      errno_exit("VIDIOC_QUERYBUF");
    }

    void *buffer = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                        fd, buf.m.offset);
    if (buffer == MAP_FAILED) {
      errno_exit("mmap");
    }

    // Queue the buffer
    if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
      errno_exit("VIDIOC_QBUF");
    }

    // Start capturing
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
      errno_exit("VIDIOC_STREAMON");
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
      errno_exit("select");
    }
    if (r == 0) {
      throw std::runtime_error("Timeout waiting for frame");
    }

    // Dequeue the buffer
    if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
      errno_exit("VIDIOC_DQBUF");
    }

    // Save the captured frame to a file using OpenCV
    if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_MJPEG) {
      std::vector<uchar> data(static_cast<unsigned char *>(buffer),
                              static_cast<unsigned char *>(buffer) +
                                  buf.bytesused);
      frame = cv::imdecode(cv::Mat(data), cv::IMREAD_COLOR);
      if (frame.empty()) {
        throw std::runtime_error("Error decoding MJPEG frame");
      }
    } else if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_YUYV) {
      cv::Mat yuyv_frame(cv::Size(fmt.fmt.pix.width, fmt.fmt.pix.height),
                         CV_8UC2, buffer);
      cv::cvtColor(yuyv_frame, frame, cv::COLOR_YUV2BGR_YUYV);
    } else {
      std::cerr << "Error: Unsupported pixel format 0x" << std::hex
                << fmt.fmt.pix.pixelformat << std::dec
                << " for direct saving.\n";
    }

    if (frame.empty()) {
      errno_exit("error: No frame data captured.");
    }

    // Stop capturing
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) < 0) {
      errno_exit("VIDIOC_STREAMOFF");
    }

    // Unmap the buffer
    if (munmap(buffer, buf.length) < 0) {
      errno_exit("munmap");
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