#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/videodev2.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

// Error handling helper function
void errno_exit(const char *s) {
  throw std::runtime_error(s + std::string(" error ") + std::to_string(errno));
}

// Function to probe the V4L2 device and list capabilities
void probeDevice(const std::string &device_path) {
  int fd = -1;
  try {
    fd = open(device_path.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) {
      errno_exit("Failed to open device for probing");
    }

    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
      errno_exit("VIDIOC_QUERYCAP during probing");
    }

    std::cerr << "Device Capabilities:\n";
    std::cerr << "  Driver Name: " << cap.driver << "\n";
    std::cerr << "  Card Name: " << cap.card << "\n";
    std::cerr << "  Bus Info: " << cap.bus_info << "\n";
    std::cerr << "  Version: " << ((cap.version >> 16) & 0xFF) << "."
              << ((cap.version >> 8) & 0xFF) << "." << (cap.version & 0xFF)
              << "\n";
    std::cerr << "  Capabilities: 0x" << std::hex << cap.capabilities
              << std::dec << "\n";
    if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)
      std::cerr << "    Supports Video Capture\n";
    if (cap.capabilities & V4L2_CAP_STREAMING)
      std::cerr << "    Supports Streaming\n";

    struct v4l2_fmtdesc fmtdesc;
    memset(&fmtdesc, 0, sizeof(fmtdesc));
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    std::cerr << "\nSupported Formats:\n";
    while (ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc) == 0) {
      std::cerr << "  Format: " << fmtdesc.description << " (Fourcc: 0x"
                << std::hex << fmtdesc.pixelformat << std::dec << ")\n";

      struct v4l2_frmsizeenum fsize;
      memset(&fsize, 0, sizeof(fsize));
      fsize.pixel_format = fmtdesc.pixelformat;
      std::cerr << "    Supported Sizes:\n";
      while (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &fsize) == 0) {
        if (fsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
          std::cerr << "      " << fsize.discrete.width << "x"
                    << fsize.discrete.height << "\n";
        } else if (fsize.type == V4L2_FRMSIZE_TYPE_CONTINUOUS) {
          std::cerr << "      Continuous: " << fsize.stepwise.min_width << "-"
                    << fsize.stepwise.max_width << " x "
                    << fsize.stepwise.min_height << "-"
                    << fsize.stepwise.max_height << "\n";
        } else if (fsize.type == V4L2_FRMSIZE_TYPE_STEPWISE) {
          std::cerr << "      Stepwise: " << fsize.stepwise.min_width << "-"
                    << fsize.stepwise.max_width << " step "
                    << fsize.stepwise.step_width << " x "
                    << fsize.stepwise.min_height << "-"
                    << fsize.stepwise.max_height << " step "
                    << fsize.stepwise.step_height << "\n";
        }
        fsize.index++;
      }
      fmtdesc.index++;
    }

  } catch (const std::runtime_error &e) {
    std::cerr << "Error during device probing: " << e.what() << std::endl;
  }
  if (fd != -1) {
    close(fd);
  }
}

// Function to capture a snapshot from a V4L2 device
bool captureSnapshot(const std::string &device_path,
                     const std::string &output_path) {
  int fd = -1;
  try {
    fd = open(device_path.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) {
      errno_exit("Failed to open device");
    }

    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
      errno_exit("VIDIOC_QUERYCAP");
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
      errno_exit("Failed to set MJPEG format");
    }
    std::cerr << "Using format: Width=" << fmt.fmt.pix.width
              << ", Height=" << fmt.fmt.pix.height << ", PixelFormat=0x"
              << std::hex << fmt.fmt.pix.pixelformat << std::dec << "\n";

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
    cv::Mat frame;
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

    if (!frame.empty() && !cv::imwrite(output_path, frame)) {
      throw std::runtime_error("Failed to save image");
    } else if (frame.empty()) {
      std::cerr << "Warning: No frame data to save.\n";
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
    std::cerr << "Error: " << e.what() << std::endl;
    if (fd != -1) {
      close(fd);
    }
    return false;
  }
}

int main() {
  std::string device = "/dev/video0";
  std::string output = "snapshot.jpg";

  probeDevice(device);

  if (captureSnapshot(device, output)) {
    std::cout << "Snapshot saved to " << output << std::endl;
  } else {
    std::cout << "Failed to capture snapshot." << std::endl;
  }
  return 0;
}