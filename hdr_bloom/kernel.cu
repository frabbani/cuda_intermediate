
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <vector>
#include <memory>

#include "bitmap.h"

/*
extern "C" {
#include "rgbe.h"
}
struct RGBE {
	int w, h;
	std::vector<float3> data;

	bool valid() const { return w && h && data.size(); }

	bool load(const char* filename) {
		char* test = (char*)filename;
		FILE* fp = fopen(filename, "rb");
		if (!fp) {
			printf("RBE::load - failed to open file \'%s\'\n", filename);
			return false;
		}

		rgbe_header_info header;
		int result = RGBE_ReadHeader(fp, &w, &h, &header);
		if (result == RGBE_RETURN_FAILURE) {
			printf("RBE::load - invalid file format for file \'%s\'\n", filename);
			return false;
		}

		data = std::vector<float3>(w * h);
		float* rgb = (float*)(data.data());

		result = RGBE_ReadPixels_RLE(fp, rgb, w, h);
		if (result == RGBE_RETURN_FAILURE) {
			w = h = 0;
			data.clear();
			printf("RBE::load - failed to read pixels from for file \'%s\'\n", filename);
			return false;
		}
		fclose(fp);
		return true;
	}
};
*/

template <typename T>
struct is_valid_cuda_var : std::false_type {};
template <> struct is_valid_cuda_var<float> : std::true_type {};
template <> struct is_valid_cuda_var<float2> : std::true_type {};
template <> struct is_valid_cuda_var<float3> : std::true_type {};
template <> struct is_valid_cuda_var<float4> : std::true_type {};
template <> struct is_valid_cuda_var<double> : std::true_type {};
template <> struct is_valid_cuda_var<double2> : std::true_type {};
template <> struct is_valid_cuda_var<double3> : std::true_type {};
template <> struct is_valid_cuda_var<double4> : std::true_type {};
template <> struct is_valid_cuda_var<int> : std::true_type {};
template <> struct is_valid_cuda_var<int2> : std::true_type {};
template <> struct is_valid_cuda_var<int3> : std::true_type {};
template <> struct is_valid_cuda_var<int4> : std::true_type {};

template<typename T>
struct CudaArray {
	static_assert(is_valid_cuda_var<T>::value, "invalid CUDA data type");

	std::vector<T> host;
	T* devptr = nullptr;

	int count() { return int(host.size()); }
	void alloc(int numValues) {
		free();
		host = std::vector<T>(numValues);
		auto status = cudaMalloc(&devptr, numValues * sizeof(T));
		if (status != cudaSuccess) {
			printf("cudaMalloc failed: %s\n", cudaGetErrorString(status));
		}
	}

	void free() {
		host.clear();
		if (devptr) {
			auto status = cudaFree(devptr);
			if (status != cudaSuccess) {
				printf("cudaFree failed: %s\n", cudaGetErrorString(status));
			}
			devptr = nullptr;
		}
	}

	void push() {
		auto status = cudaMemcpy(devptr, host.data(), count() * sizeof(T), cudaMemcpyHostToDevice);
		if (status != cudaSuccess) {
			printf("cudaMemcpy failed: %s\n", cudaGetErrorString(status));
		}
	}

	void pull() {
		auto status = cudaMemcpy(host.data(), devptr, count() * sizeof(T), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			printf("cudaMemcpy failed: %s\n", cudaGetErrorString(status));
		}
	}
};

#define CLAMP(v, a, b)  {v = v < (a) ? (a) : v > (b) ? (b) : v; }

#define IMAGE_DIM   1024
#define BLOCK_DIM   16
#define WARP_SIZE	32

//sum = 256
__constant__ float kGauss5[5][5] = {
		{ 1,  4,  6,  4, 1 },
		{ 4, 16, 24, 16, 4 },
		{ 6, 24, 36, 24, 6 },
		{ 4, 16, 24, 16, 4 },
		{ 1,  4,  6,  4, 1 }
};

//sum = 4783
__constant__ float kGauss9[9][9] = {
		{  0,  0,  1,  2,  2,  2,  1,  0,  0 },
		{  0,  3, 13, 22, 26, 22, 13,  3,  0 },
		{  1, 13, 59, 97,112, 97, 59, 13,  1 },
		{  2, 22, 97,159,184,159, 97, 22,  2 },
		{  2, 26,112,184,213,184,112, 26,  2 },
		{  2, 22, 97,159,184,159, 97, 22,  2 },
		{  1, 13, 59, 97,112, 97, 59, 13,  1 },
		{  0,  3, 13, 22, 26, 22, 13,  3,  0 },
		{  0,  0,  1,  2,  2,  2,  1,  0,  0 }
};

//sum = 19307
__constant__ float kGauss13[13][13] = {
	{ 0,  0,  0,  1,  2,  3,  3,  3,  2,  1,  0,  0,  0 },
	{ 0,  1,  3,  7, 11, 14, 16, 14, 11,  7,  3,  1,  0 },
	{ 0,  3, 10, 22, 33, 41, 46, 41, 33, 22, 10,  3,  0 },
	{ 1,  7, 22, 49, 72, 89,100, 89, 72, 49, 22,  7,  1 },
	{ 2, 11, 33, 72,107,132,148,132,107, 72, 33, 11,  2 },
	{ 3, 14, 41, 89,132,163,183,163,132, 89, 41, 14,  3 },
	{ 3, 16, 46,100,148,183,206,183,148,100, 46, 16,  3 },
	{ 3, 14, 41, 89,132,163,183,163,132, 89, 41, 14,  3 },
	{ 2, 11, 33, 72,107,132,148,132,107, 72, 33, 11,  2 },
	{ 1,  7, 22, 49, 72, 89,100, 89, 72, 49, 22,  7,  1 },
	{ 0,  3, 10, 22, 33, 41, 46, 41, 33, 22, 10,  3,  0 },
	{ 0,  1,  3,  7, 11, 14, 16, 14, 11,  7,  3,  1,  0 },
	{ 0,  0,  0,  1,  2,  3,  3,  3,  2,  1,  0,  0,  0 }
};

__device__ __constant__ float kExposure;
__device__ __constant__ float kBloomThreshold;
__device__ __constant__ float kBloomSoftness;
__device__ __constant__ float3 kBloomFactor;

void setCudaExposure(float exposure) {
	CLAMP(exposure, 0.0f, 1.0f);
	//cudaMemcpyToSymbol("kBloomThreshold", &threshold, sizeof(float));
	cudaMemcpyToSymbol(kExposure, &exposure, sizeof(float));
}

void setCudaBloomThreshold(float threshold) {
	CLAMP(threshold, 0.0f, 1.0f);
	//cudaMemcpyToSymbol("kBloomThreshold", &threshold, sizeof(float));
	cudaMemcpyToSymbol(kBloomThreshold, &threshold, sizeof(float));
}

void setCudaBloomSoftness(float softness) {
	CLAMP(softness, 0.0f, 1.0f);
	cudaMemcpyToSymbol(kBloomSoftness, &softness, sizeof(float));
}

void setCudaBloomFactor(float3 factor) {
	float sum = factor.x;
	sum += factor.y;
	sum += factor.z;
	if (sum > 0.0) {
		factor.x /= sum;
		factor.y /= sum;
		factor.z /= sum;
	}
	cudaMemcpyToSymbol(kBloomFactor, &factor, sizeof(float3));
}

__global__ void test_kernel(float3* output) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	output[y * IMAGE_DIM + x] = { kBloomThreshold, kBloomThreshold, kBloomThreshold };
}

__device__ __host__ float to_linear(float c) {
	return  (c <= 0.04045f) ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
}

__device__ __host__ float to_srgb(float c) {
	return (c <= 0.0031308f) ? 12.92f * c : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

__device__ __host__ float3 to_linear(float3 c) {
	c.x = to_linear(c.x);
	c.y = to_linear(c.y);
	c.z = to_linear(c.z);
	return c;
}

__device__ __host__ float3 to_standard(float3 c) {
	c.x = to_srgb(c.x);
	c.y = to_srgb(c.y);
	c.z = to_srgb(c.z);
	return c;
}

__device__ __host__ float luminance(float3 c) {
	float3 lin_c = to_linear(c);
	return 0.2126f * lin_c.x + 0.7152f * lin_c.y + 0.0722f * lin_c.z;
}

__device__ float bloom_weight(float lum) {
	lum -= kBloomThreshold;
	lum = lum < 0.0 ? 0.0 : lum;
	float w = lum / kBloomSoftness;
	w = w > 1.0 ? 1.0 : w;
	return w;
}

__device__ __host__ float3 f3muls(float s, float3 v) {
	v.x *= s;
	v.y *= s;
	v.z *= s;
	return v;
}

__device__ __host__ float3 f3add(float3 u, float3 v) {
	u.x += v.x;
	u.y += v.y;
	u.z += v.z;
	return u;
}

__device__ __host__ float3 f3madd(float s, float3 u, float3 v) {
	u.x = s * u.x + v.x;
	u.y = s * u.y + v.y;
	u.z = s * u.z + v.z;
	return u;
}

__global__ void bloom_extract_kernel(const float3* input, float3* output, int w, int h) {
	const int tile_dim = BLOCK_DIM * 2;
	const int halo_dim = BLOCK_DIM / 2;
	const float3 zero = { 0.0f, 0.0f, 0.0f };

	auto input_sample = [&](int x, int y)
		{
			CLAMP(x, 0, w - 1);
			CLAMP(y, 0, h - 1);
			return input[y * w + x];
		};

	__shared__ float3 tile[tile_dim * tile_dim];
	{
		int x = 2 * threadIdx.x;
		int y = 2 * threadIdx.y;
		int gx = blockDim.x * blockIdx.x + x - halo_dim;
		int gy = blockDim.y * blockIdx.y + y - halo_dim;

		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++) {
				int idx = (y + i) * tile_dim + x + j;
				float3 c = to_linear(input_sample(gx + j, gy + i));
				float weight = bloom_weight(luminance(c));
				tile[idx] = f3muls(weight, c);
			}
		__syncthreads();
	}
	auto tile_sample = [&](int x, int y) -> float3&
		{
			x += halo_dim;
			y += halo_dim;
			//CLAMP(x, 0, tile_dim - 1);
			//CLAMP(y, 0, tile_dim - 1);
			return tile[y * tile_dim + x];
		};

	int x = threadIdx.x;
	int y = threadIdx.y;

	float3 core_spread, medium_spread, wide_spread;

	core_spread = zero;
	for (int i = -2; i <= 2; i++)
		for (int j = -2; j <= 2; j++)
			core_spread = f3add(core_spread, f3muls(kGauss5[i + 2][j + 2], tile_sample(x + j, y + i)));
	core_spread = f3muls(1.0f / 256.0f, core_spread);

	for (int i = -4; i <= 4; i++) 
		for (int j = -4; j <= 4; j++)
			medium_spread = f3add(medium_spread, f3muls(kGauss9[i + 4][j + 4], tile_sample(x + j, y + i)));
	medium_spread = f3muls(1.0f / 4783.0f, medium_spread);

	wide_spread = zero;
	for (int i = -6; i <= 6; i++)
		for (int j = -6; j <= 6; j++)
			wide_spread = f3add(wide_spread, f3muls(kGauss13[i + 6][j + 6], tile_sample(x + j, y + i)));
	wide_spread = f3muls(1.0f / 19307.0f, wide_spread);

	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int gy = blockIdx.y * blockDim.y + threadIdx.y;

	output[gy * w + gx] = f3madd( kBloomFactor.x, core_spread, f3madd(kBloomFactor.y, medium_spread, f3muls(kBloomFactor.y, wide_spread)));
}

__global__ void bloom_apply_kernel(const float3* input, float3* bloom, float3 *output, int w, int h) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * w + x;
	float3 v = to_linear(input[idx]);

	float lum = luminance(v);
	float lum_tmap = lum / (lum + 1.0f);	// Reinhard  tone-mapping
	v = f3muls(lum_tmap / lum, v);
	v = f3add(v, bloom[idx]);
	CLAMP(v.x, 0.0f, 1.0f);
	CLAMP(v.y, 0.0f, 1.0f);
	CLAMP(v.z, 0.0f, 1.0f);
	output[idx] = to_standard(v);
}

__global__ void log_average_kernel(const float3 *input, float *output) {
	// assume block size = warp size, input size = number of blocks * warp size, and output size = input size / warp size, or block size

	int lane = (threadIdx.x & 0x1f);
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	// parallel tree reduction - an insane technique for using a binary tree structure for summing.
	// also known as logarithmic reduction

	//float log_sum = logf(1e-4f + luminance(input[i]));
	//for (int offset = 16; offset > 0; offset /= 2)
	//	log_sum += __shfl_down_sync(0xffffffff, log_sum, offset);	// shuffle down reads from higher indices
	//if (lane == 0)
	//	output[blockIdx.x] = log_sum / 32.0;

	// the pedagogically clear, manually staged version. 
	float log_sums[6];	//0,1,2,4,8,16
	int step = 1;
	log_sums[0] = logf(1e-4f + luminance(input[i]));
	for (int offset = 1; offset <= 16; offset *= 2, step++) {
		log_sums[step] = log_sums[step - 1] + __shfl_sync(0xffffffff, log_sums[step - 1], lane + offset);
	}
	if(lane == 0)
		output[blockIdx.x] = log_sums[step-1] / 32.0;
}

__global__ void log_average_kernel(const float* input, float* output) {
	// assume block size = warp size, input size = number of blocks * warp size, and output size = input size / warp size
	int lane = threadIdx.x & 0x1f;
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	// parallel tree reduction - an insane technique for using a binary tree structure for summing.
	// also known as logarithmic reduction
	float log_sum = input[i];
	for (int offset = 1; offset <= 16; offset *= 2)
		log_sum += __shfl_up_sync(0xffffffff, log_sum, offset);	// shuffle up reads from lower indices
	if (lane == 31)
		output[blockIdx.x] = log_sum / 32.0;
}


bool loadBitmap(CudaArray<float3>& input, int& w, int& h, const char* bitmapName) {
	std::vector<uint8_t> image;
	importBMP(image, w, h, bitmapName);
	if (!image.size())
		return false;
	if (!image.size() || w & 0x1f || h & 0x1f) {
		printf("%s - bitmap image '%s' dimensions are not a multiple of 32\n");
		return false;
	}
	input.alloc(w * h);
	for (int i = 0; i < w * h; i++) {
		input.host[i].z = float(image[i * 3 + 0]) / 255.0;
		input.host[i].y = float(image[i * 3 + 1]) / 255.0;
		input.host[i].x = float(image[i * 3 + 2]) / 255.0;
	}
	input.push();
}

void saveBitmap(CudaArray<float3>& input, int w, int h, const char* bitmapName) {
	input.pull();
	std::vector<uint8_t> pixels(w * h * 3);
	for (int i = 0; i < w * h; i++) {
		pixels[i * 3 + 0] = uint8_t(input.host[i].z * 255.0);
		pixels[i * 3 + 1] = uint8_t(input.host[i].y * 255.0);
		pixels[i * 3 + 2] = uint8_t(input.host[i].x * 255.0);
	}
	exportBMP(pixels, w, h, bitmapName);
}

bool createDownsample(CudaArray<float>& output, int& size) {
	output.free();
	if (size >= WARP_SIZE) {
		size /= WARP_SIZE;
		output.alloc(size);
		return true;
	}
	return false;
}

int main()
{
	printf("hello world!\n");

	int n = 0;
	cudaGetDeviceCount(&n);
	printf("%d CUDA device(s)\n", n);
	auto status = cudaSetDevice(0);
	if (status != cudaSuccess) {
		printf("cudaSetDevice failed: %s\n", cudaGetErrorString(status));
		return 0;
	}

	int w, h;
	CudaArray<float3> input, bloom, output;
	int numds = 0;
	CudaArray<float> downs[8];	// need 6 for 8k by 8k image
	dim3 block, grid;
	float lumAvg = 0.0f;

	cudaEvent_t e0, e1;
	float ms = 0.0f;

	if (!loadBitmap(input, w, h, "input.bmp")) {
		cudaDeviceReset();
		return 0;
	}
	bloom.alloc(w * h);
	output.alloc(w * h);
	int size = w * h;
	while (createDownsample(downs[numds], size)) {
		numds++;
	}

	cudaEventCreate(&e0);
	cudaEventCreate(&e1);
	cudaDeviceSynchronize();

	cudaEventRecord(e0);
	block = dim3(WARP_SIZE, 1, 1);
	grid = dim3(input.count() / WARP_SIZE, 1, 1);
	int ds = 0;
	log_average_kernel << <grid, block >> > (input.devptr, downs[ds].devptr);
	while (downs[ds].count() >= WARP_SIZE) {
		//downs[ds].pull();
		//printf("size @ %d: %d (value %f)\n", ds, downs[ds].count(), to_srgb(expf(downs[ds].host[0])));
		grid = dim3(downs[ds+1].count(), 1, 1);
		log_average_kernel << <grid, block >> > (downs[ds].devptr, downs[ds+1].devptr);
		ds++;
	}
	cudaEventRecord(e1);
	cudaEventSynchronize(e1);
	cudaEventElapsedTime(&ms, e0, e1);
	downs[ds].pull();
	for (int i = 0; i < downs[ds].count(); i++)
		lumAvg += downs[ds].host[i];
	lumAvg /= float(downs[ds].count());
	lumAvg = to_srgb(expf(lumAvg));
	printf("scene's average luminance: %f (after %f ms)\n", lumAvg, ms);

	block = dim3(BLOCK_DIM, BLOCK_DIM);
	grid = dim3(w / BLOCK_DIM, h / BLOCK_DIM);

	setCudaBloomThreshold(to_linear(0.1) * lumAvg);
	setCudaBloomSoftness(0.4);
	setCudaBloomFactor({0.333f, 0.333f, 0.333f});
	cudaEventRecord(e0);
	bloom_extract_kernel << <grid, block >> > (input.devptr, bloom.devptr, w, h);
	bloom_apply_kernel << <grid, block >> > (input.devptr, bloom.devptr, output.devptr, w, h);
	cudaEventRecord(e1);
	cudaEventSynchronize(e1);
	cudaEventElapsedTime(&ms, e0, e1);
	printf("total time: %f ms\n", ms);
	saveBitmap(bloom, w, h, "linear_bloom.bmp");
	saveBitmap(output, w, h, "tone_mapped.bmp");


	input.free();
	output.free();
	for (int i = 0; i < numds; i++)
		downs[i].free();
	cudaEventDestroy(e0);
	cudaEventDestroy(e1);

	status = cudaDeviceReset();
	if (status != cudaSuccess) {
		printf("cudaDeviceReset failed: %s\n", cudaGetErrorString(status));
	}

	printf("goodbye!\n");
	return 0;
}
