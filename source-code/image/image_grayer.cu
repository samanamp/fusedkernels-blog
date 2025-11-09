#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <cuda_runtime.h>
#include "stb_image_write.h"

__global__ void grayScaler(unsigned char *in_d, unsigned char *out_d, int width, int height, int channels)
{
    int x_i = blockIdx.x * blockDim.x + threadIdx.x;
    int y_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_i < width && y_i < height)
    {
        int pixeloffset = (y_i * width + x_i);
        int grayoffset = pixeloffset * (channels - 2);
        int color_offset = pixeloffset * channels;
        unsigned char r = in_d[color_offset];
        unsigned char g = in_d[color_offset + 1];
        unsigned char b = in_d[color_offset + 2];
        out_d[grayoffset] = 0.21f * r + 0.71f * g + 0.07f * b;
        if (channels == 4)
            out_d[grayoffset + 1] = in_d[color_offset + 3];
    }
}

int main()
{
    int width, height, channels = 4;
    // loads image forcing 3-4 channels. doesn't break the code
    unsigned char *data = stbi_load("crocodile.png", &width, &height, &channels, channels);
    // uses all channels
    // unsigned char *data = stbi_load("crocodile.png", &width, &height, &channels, 0);
    if (!data)
    {
        std::cerr << "Failed to load image\n";
        return 1;
    }

    // Access pixel (row=y, col=x)
    int x = width / 2, y = height / 2;
    unsigned char *pixel = data + (y * width + x) * channels;
    int size = (width * height * channels) * sizeof(char);
    printf("Channels=%d .. R=%d, G=%d, B=%d\n", channels, pixel[0], pixel[1], pixel[2]);
    printf("image info: width=%d, height=%d, size=%d \n", width, height, size);

    unsigned char *d_data;
    cudaError_t cuda_status = cudaMalloc((void **)&d_data, size);
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc input failed! Error: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    cuda_status = cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpyHostToDevice failed! Error: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }
    // allocate space for output
    int output_channels = (channels == 4) ? 2 : 1;
    int output_size = width * height * output_channels;
    unsigned char *out_h = (unsigned char *)malloc(output_size);
    unsigned char *out_d;
    cuda_status = cudaMalloc((void **)&out_d, output_size);
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc output failed! Error: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }
    // calculate kernel params
    // We put numbers as 16.0 to signal the compiler to use float
    dim3 gridSize(ceil(width / 16.0), ceil(height / 16.0), 1);
    dim3 blockSize(16, 16, 1);
    // call kernel
    grayScaler<<<gridSize, blockSize>>>(d_data, out_d, width, height, channels);
    // copy output to host
    cuda_status = cudaMemcpy(out_h, out_d, output_size, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc output failed! Error: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    unsigned char *n_pixel = out_h + (y * width + x);
    printf("Gray=%d \n", n_pixel[0]);

    // Save the grayscale image as PNG (1-2 channel)
    stbi_write_png("crocodile_gray_transparent.png", width, height,
                   output_channels, out_h, width * output_channels);
    printf("Saved grayscale image to crocodile_gray.png\n");

    cuda_status = cudaFree(d_data);
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "cudaFree failed! Error: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }
    stbi_image_free(data);
    return 0;
}
