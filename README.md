# CUDA Background Video Replacement

This project implements real-time background replacement using CUDA and OpenCV. It leverages a custom CUDA kernel to detect white (or near-white) pixels in a webcam feed and replaces them with frames from a video file. This mimics green screen effects but instead targets users with plain white or bright backdrops.

---

## Features

- Real-time video stream processing using CUDA
- HSV-based white detection for chroma keying without a green screen
- Background replacement using a looping video
- GPU-accelerated image processing via CUDA kernels

---

## Improvement Highlights
-  Replaced static green screen logic with adaptive white background detection.
-  Improved background removal performance using CUDA kernel parallelism.
-  Enhanced visual quality with transparency-aware compositing.
-  Support for video background replacement (instead of static images).
-  Real-time frame synchronization between webcam and video.

---

## Requirements

- Windows with Visual Studio 2022
- CUDA Toolkit 12.5 or later
- OpenCV 4.1.2 (or compatible prebuilt version)
- NVIDIA GPU with compute capability ≥ 5.0

---

## Directory Structure


├── CMakeLists.txt
├── main.cu
├── .gitignore
└── bg.mp4 (your looping video)

---

## How to Build

1. Make sure `nvcc` is installed. If not, install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

2. Open a command prompt in the project directory and run:

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe" main.cu -o chroma.exe ^
-I"C:\opencv\build\include" ^
-L"C:\opencv\build\x64\vc17\lib" -lopencv_world4120

> Adjust the paths above to match your OpenCV installation.

---

## How to Run

Place a video file (e.g., `bg.mp4`) in the same folder as `chroma.exe`. Then:

stage18.exe

The app will:

- Open your webcam feed
- Detect white regions in the frame (background)
- Replace them in real-time with looping video

---

## Notes

- If the webcam doesn’t open, ensure it’s not in use by another application.
- The `main.cu` code supports editing HSV thresholds for better masking accuracy.
- You can customize the overlay color by changing the blending section in the CUDA kernel.

---

## Customization

To change the white detection sensitivity, modify this section in `main.cu`:

```cpp
// Define near-white HSV bounds
int h_min = 0, h_max = 180;
int s_min = 0, s_max = 30;
int v_min = 200, v_max = 255;
```

To change overlay behavior, adjust the blending logic in the CUDA kernel.

---

## License

This project is under the MIT License. You may freely modify and distribute it.

---

## Author

Created by Blastermania (Tanish Dey)
