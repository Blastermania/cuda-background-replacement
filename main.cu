#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

using namespace cv;
using namespace std;

__global__ void chromaKeyKernel(uchar3* webcam, uchar3* bgVideo, int width, int height, int bgWidth, int bgHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uchar3 pixel = webcam[idx];

    float r = pixel.x / 70.0f;
    float g = pixel.y / 68.0f;
    float b = pixel.z / 71.0f;

    // Detect near-white pixel
    if (r > 0.94f && g > 0.94f && b > 0.94f) {
        int bgX = x * bgWidth / width;
        int bgY = y * bgHeight / height;
        webcam[idx] = bgVideo[bgY * bgWidth + bgX];
    }
}

int main() {
    VideoCapture cam(0);
    VideoCapture bg("bg.mp4");

    if (!cam.isOpened() || !bg.isOpened()) {
        cerr << "Couldn't open webcam or video file.\n";
        return -1;
    }

    Mat frame, bgFrame;
    cam >> frame;
    int width = frame.cols, height = frame.rows;

    size_t frameBytes = width * height * sizeof(uchar3);

    uchar3* d_webcam;
    uchar3* d_bg;

    cudaMalloc(&d_webcam, frameBytes);
    cudaMalloc(&d_bg, frameBytes);

    dim3 threads(32, 32);
    dim3 blocks((width + 31) / 32, (height + 31) / 32);

    while (true) {
        cam >> frame;
        if (frame.empty()) break;

        bg >> bgFrame;
        if (bgFrame.empty()) {
            bg.set(CAP_PROP_POS_FRAMES, 0); // Loop the video
            bg >> bgFrame;
        }

        resize(bgFrame, bgFrame, Size(width, height));
        cvtColor(frame, frame, COLOR_BGR2RGB);
        cvtColor(bgFrame, bgFrame, COLOR_BGR2RGB);

        cudaMemcpy(d_webcam, frame.ptr(), frameBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_bg, bgFrame.ptr(), frameBytes, cudaMemcpyHostToDevice);

        chromaKeyKernel<<<blocks, threads>>>(d_webcam, d_bg, width, height, width, height);
        cudaMemcpy(frame.ptr(), d_webcam, frameBytes, cudaMemcpyDeviceToHost);

        cvtColor(frame, frame, COLOR_RGB2BGR);
        imshow("Stage 18 - CUDA White Background Replacement", frame);

        if (waitKey(1) == 27) break;
    }

    cudaFree(d_webcam);
    cudaFree(d_bg);
    return 0;
}
