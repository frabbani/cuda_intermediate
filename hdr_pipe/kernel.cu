
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <vector>
#include <array>

#include "bitmap.h"

extern "C" {
#include "rgbe.h"
}

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

struct CudaEventPair {
	cudaEvent_t e0, e1;
	cudaStream_t stream = nullptr;

	void init() {
		cudaEventCreate(&e0);
		cudaEventCreate(&e1);
		stream = nullptr;
	}

	void setStream(cudaStream_t stream) {
		this->stream = stream;
	}

	void start() {
		cudaEventRecord(e0, stream);
	}

	void finish() {
		cudaEventRecord(e1, stream);
	}

	void sync() {
		cudaEventSynchronize(e1); // 'cudaEventSynchronize(e0)' not needed
	}

	float elapsedMs() {
		float ms = 0.0f;
		cudaEventElapsedTime(&ms, e0, e1);
		return ms;
	}

	void destroy() {
		cudaEventDestroy(e0);
		cudaEventDestroy(e1);
		stream = nullptr;
	}
};

template<typename T>
struct CudaArray {
	static_assert(is_valid_cuda_var<T>::value, "invalid CUDA data type");

	std::vector<T> host;
	T* devptr = nullptr;

	int count() const { return int(host.size()); }
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

/*
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
*/

__device__ __constant__ float kExposure;
__device__ __constant__ float kGrayKey;
__device__ __constant__ float kBloomAlpha;
__device__ __constant__ float4 kBloomFactor;

void setCudaExposure(float exposure) {
	cudaMemcpyToSymbol(kExposure, &exposure, sizeof(float));
}

void setCudaGrayKey(float grayKey) {
	cudaMemcpyToSymbol(kGrayKey, &grayKey, sizeof(float));
}

void setCudaBloomFactor(float4 factor) {
	float sum = factor.x;
	sum += factor.y;
	sum += factor.z;
	sum += factor.w;
	if (sum > 0.0) {
		factor.x /= sum;
		factor.y /= sum;
		factor.z /= sum;
		factor.w /= sum;
	}
	cudaMemcpyToSymbol(kBloomFactor, &factor, sizeof(float4));
}

void setCudaBloomAlpha(float alpha) {
	alpha = alpha < 0.0 ? 0.0 : alpha;
	cudaMemcpyToSymbol(kBloomAlpha, &alpha, sizeof(float));
}

__global__ void test_kernel(float3* output) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	output[y * IMAGE_DIM + x] = { 0.5f, 0.5f, 0.5f };
}

__device__ __host__ float to_linear(float c) {
	return  (c <= 0.04045f) ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
}

__device__ __host__ float to_standard(float c) {
	return (c <= 0.0031308f) ? 12.92f * c : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

__device__ __host__ float3 to_linear(float3 c) {
	c.x = to_linear(c.x);
	c.y = to_linear(c.y);
	c.z = to_linear(c.z);
	return c;
}

__device__ __host__ float3 to_srgb(float3 c) {
	c.x = to_standard(c.x);
	c.y = to_standard(c.y);
	c.z = to_standard(c.z);
	return c;
}

__device__ __host__ float luminance(float3 c_lin) {
	return 0.2126f * c_lin.x + 0.7152f * c_lin.y + 0.0722f * c_lin.z;
}

__device__ __host__ float luminance_srgb(float3 c) {
	return luminance(to_linear(c));
}

__device__ float bloom_attenuation(float lum) {
	return exp(-kBloomAlpha / (lum + 0.0001));
}

__device__ __host__ __forceinline__ float3 f3muls(float s, float3 v) {
	v.x *= s;
	v.y *= s;
	v.z *= s;
	return v;
}

__device__ __host__ __forceinline__ float3 f3add(float3 u, float3 v) {
	u.x += v.x;
	u.y += v.y;
	u.z += v.z;
	return u;
}

__device__ __host__ __forceinline__ float3 f3sub(float3 u, float3 v) {
	u.x -= v.x;
	u.y -= v.y;
	u.z -= v.z;
	return u;
}

__device__ __host__ __forceinline__ float3 f3madd(float s, float3 u, float3 v) {
	u.x = s * u.x + v.x;
	u.y = s * u.y + v.y;
	u.z = s * u.z + v.z;
	return u;
}

__device__ __host__ __forceinline__ float3 f3lerp(float3 u, float3 v, float s) {
	return f3madd(s, f3sub(v, u), u);
}

__device__ __host__ float3 sample_normalized(float2 uv, const float3* input, int w, int h) {
	//CLAMP(uv.x, 0.0, 1.0);
	//CLAMP(uv.y, 0.0, 1.0);

	float sx = float(w);
	float sy = float(h);
	float x = uv.x * sx;
	float y = uv.y * sy;

	int x1 = int(x);
	int y1 = int(y);
	int x2 = x1 + 1;
	int y2 = y1 + 1;
	//CLAMP(x1, 0, w - 1);
	CLAMP(x2, 0, w - 1);
	//CLAMP(y1, 0, h - 1);
	CLAMP(y2, 0, h - 1);

	float3 taps[2][2], t0, t1;
	taps[0][0] = input[y1 * w + x1];
	taps[0][1] = input[y1 * w + x2];
	taps[1][0] = input[y2 * w + x1];
	taps[1][1] = input[y2 * w + x2];

	float s = x - float(x1);
	float t = y - float(y1);
	//CLAMP(s, 0.0, 1.0);
	//CLAMP(t, 0.0, 1.0);
	t0 = f3lerp(taps[0][0], taps[0][1], s);
	t1 = f3lerp(taps[1][0], taps[1][1], s);
	return f3lerp(t0, t1, t);
}

/*
__global__ void mip_kernel(const float3* input, int w, int h, float3* output, bool isLinear = true) {
	const float3 zero = { 0.0f, 0.0f, 0.0f };
	auto sample_input = [&](int x, int y)
		{
			x *= 2;
			y *= 2;
			float3 out = zero;
			out = f3madd(0.25, input[(y + 0) * w + (x + 0)], out);
			out = f3madd(0.25, input[(y + 0) * w + (x + 1)], out);
			out = f3madd(0.25, input[(y + 1) * w + (x + 0)], out);
			out = f3madd(0.25, input[(y + 1) * w + (x + 1)], out);
			return out;
		};
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int gy = blockIdx.y * blockDim.y + threadIdx.y;
	output[gy * (w / 2) + gx] = sample_input(gx, gy);
}


__global__ void bloom_extract_kernel(const float3* input, float3* output, int w, int h, bool isLinear = true) {
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
				float3 c = input_sample(gx + j, gy + i);
				c = isLinear ? c : to_linear(c);
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

	output[gy * w + gx] = f3madd(kBloomFactor.x, core_spread, f3madd(kBloomFactor.y, medium_spread, f3muls(kBloomFactor.z, wide_spread)));
}
*/

__global__ void bloom_apply_kernel(const float3* input, const float3* bloom, float3* output, int w, int h, float lumAvg, bool isLinear = true) {
	const float3 zero = { 0.0f, 0.0f, 0.0f };

	auto reinhard = [](float lum)
		{
			return lum / (lum + 1.0f);
		};
	auto key_scaled_reinhard = [](float lum, float lum_avg)
		{
			float scaled_lum = kGrayKey / lum_avg * lum;
			return scaled_lum / (scaled_lum + 1.0f);
		};


	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * w + x;
	float3 v = isLinear ? input[idx] : to_linear(input[idx]);
	v = f3add(bloom[idx], v);
	v = f3muls(exp2f(kExposure), v);
	float lum = luminance(v);
	float lum_tonemap = key_scaled_reinhard(lum, lumAvg);

	v = lum < 1e-4 ? zero : f3muls(lum_tonemap / lum, v);
	v = to_srgb(v);
	CLAMP(v.x, 0.0f, 1.0f);
	CLAMP(v.y, 0.0f, 1.0f);
	CLAMP(v.z, 0.0f, 1.0f);
	output[idx] = v;


}

// assume block size = warp size, input size = number of blocks * warp size, and output size = input size / warp size
__global__ void log_average_kernel(const float3* input, float* output, bool isLinear = true) {
	// assume block size = warp size, input size = number of blocks * warp size, and output size = input size / warp size, or block size

	int lane = (threadIdx.x & 0x1f);
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	float3 v = isLinear ? input[i] : to_linear(input[i]);
	// parallel tree reduction - an insane technique for using a binary tree structure for summing.
	// also known as logarithmic reduction

	//float log_sum = logf(1e-4f + luminance(input[i], isLinear));
	//for (int offset = 16; offset > 0; offset /= 2)
	//	log_sum += __shfl_down_sync(0xffffffff, log_sum, offset);	// shuffle down reads from higher indices
	//if (lane == 0)
	//	output[blockIdx.x] = log_sum / 32.0;

	// the pedagogically clear, manually staged version. 
	float log_sums[6];	//0,1,2,4,8,16
	int step = 1;
	log_sums[0] = logf(1e-4f + luminance(v));
	for (int offset = 1; offset <= 16; offset *= 2, step++) {
		log_sums[step] = log_sums[step - 1] + __shfl_sync(0xffffffff, log_sums[step - 1], lane + offset);
	}
	if (lane == 0)
		output[blockIdx.x] = log_sums[step - 1] / 32.0;
}

// assume block size = warp size, input size = number of blocks * warp size, and output size = input size / warp size
__global__ void average_kernel(const float* input, float* output) {
	int lane = threadIdx.x & 0x1f;
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	// parallel tree reduction - an insane technique for using a binary tree structure for summing.
	// also known as logarithmic reduction
	float sum = input[i];
	for (int offset = 1; offset <= 16; offset *= 2)
		sum += __shfl_up_sync(0xffffffff, sum, offset);	// shuffle up reads from lower indices
	if (lane == 31)
		output[blockIdx.x] = sum / 32.0;
}

//generate first level blur
__global__ void bloom_extract_and_blur_5x5_kernel(const float3* input, float3* output, int w, int h, bool isLinear = true) {
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
				float3 c = input_sample(gx + j, gy + i);
				c = isLinear ? c : to_linear(c);
				float lum = luminance(c);
				tile[idx] = f3muls(bloom_attenuation(lum), c);
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

	float3 spread;

	spread = zero;
	for (int i = -2; i <= 2; i++)
		for (int j = -2; j <= 2; j++)
			spread = f3add(spread, f3muls(kGauss5[i + 2][j + 2], tile_sample(x + j, y + i)));
	spread = f3muls(1.0f / 256.0f, spread);

	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int gy = blockIdx.y * blockDim.y + threadIdx.y;

	output[gy * w + gx] = spread;
}

__global__ void mip_and_blur_5x5_kernel(const float3* input, int w, int h, float3* output) {
	const int tile_dim = BLOCK_DIM * 2;
	const int halo_dim = BLOCK_DIM / 2;
	const float3 zero = { 0.0f, 0.0f, 0.0f };
	auto sample_input = [&](int x, int y)
		{
			x *= 2;
			y *= 2;
			float3 out = zero;
			out = f3madd(0.25, input[(y + 0) * w + (x + 0)], out);
			out = f3madd(0.25, input[(y + 0) * w + (x + 1)], out);
			out = f3madd(0.25, input[(y + 1) * w + (x + 0)], out);
			out = f3madd(0.25, input[(y + 1) * w + (x + 1)], out);
			return out;
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
				tile[idx] = sample_input(gx + j, gy + i);
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

	float3 spread;

	spread = zero;
	for (int i = -2; i <= 2; i++)
		for (int j = -2; j <= 2; j++)
			spread = f3add(spread, f3muls(kGauss5[i + 2][j + 2], tile_sample(x + j, y + i)));
	spread = f3muls(1.0f / 256.0f, spread);

	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int gy = blockIdx.y * blockDim.y + threadIdx.y;

	output[gy * w / 2 + gx] = spread;
}

__global__ void combine_mips_and_blur_5x5_kernel(float3* output, int w, int h, const float3* mip0, const float3* mip1, const float3* mip2, const float3* mip3) {
	const int tile_dim = BLOCK_DIM * 2;
	const int halo_dim = BLOCK_DIM / 2;
	const float3 zero = { 0.0f, 0.0f, 0.0f };
	float inv_w = 1.0 / float(w);
	float inv_h = 1.0 / float(h);
	int w_div_2 = w / 2;
	int w_div_4 = w / 4;
	int w_div_8 = w / 8;
	int w_div_16 = w / 16;
	int h_div_2 = h / 2;
	int h_div_4 = h / 4;
	int h_div_8 = h / 8;
	int h_div_16 = h / 16;
	auto sample = [&](int x, int y)
		{
			CLAMP(x, 0, w - 1);
			CLAMP(y, 0, h - 1);
			float2 uv = { float(x) * inv_w, float(y) * inv_h };
			float3 c0 = sample_normalized(uv, mip0, w_div_2, h_div_2);
			float3 c1 = sample_normalized(uv, mip1, w_div_4, h_div_4);
			float3 c2 = sample_normalized(uv, mip2, w_div_8, h_div_8);
			float3 c3 = sample_normalized(uv, mip3, w_div_16, h_div_16);

			return f3madd(kBloomFactor.x, c0, f3madd(kBloomFactor.y, c1, f3madd(kBloomFactor.z, c2, f3muls(kBloomFactor.w, c3))));
			//float4 a = { 0.25f, 0.25f, 0.25f, 0.25f };
			//return f3madd(a.x, c0, f3madd(a.y, c1, f3madd(a.z, c2, f3muls(a.w, c3))));

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
				tile[idx] = sample(gx + j, gy + i);
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

	float3 spread;

	spread = zero;
	for (int i = -2; i <= 2; i++)
		for (int j = -2; j <= 2; j++)
			spread = f3add(spread, f3muls(kGauss5[i + 2][j + 2], tile_sample(x + j, y + i)));
	spread = f3muls(1.0f / 256.0f, spread);

	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int gy = blockIdx.y * blockDim.y + threadIdx.y;

	output[gy * w + gx] = spread;
}


namespace globals {
	struct RGBEImage {
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
	
		void upscale( int wNew, int hNew ) {
			std::vector<float3> newData(wNew * hNew);
			for (int y = 0; y < hNew; y++) {
				for (int x = 0; x < wNew; x++) {
					int i = y * wNew + x;
					float2 uv;
					uv.x = float(x) / float(wNew);
					uv.y = float(y) / float(hNew);
					newData[i] = sample_normalized(uv, data.data(), w, h);
				}
			}
			data = std::move(newData);
			w = wNew;
			h = hNew;
		}
	};

	//image dimensions must be a multiple of 128, so the lowest mipmap is a multiple of 16
	bool loadHDR(CudaArray<float3>& input, int& w, int& h, const char* hdrFileName) {
		RGBEImage image;
		if (!image.load(hdrFileName))
			return false;
		if (image.w < 128 || image.h < 128) {
			printf("%s - image size is to small (width & height must be 128 or greater)\n");
			return false;
		}

		w = image.w;
		h = image.h;
		if (image.w & 0x7f || image.h & 0x7f) {
			printf("%s - heads up! HDR image '%s' dimensions are not a multiples of 128, upscaling it\n", __FUNCTION__, hdrFileName);
			if (w & 0x7f) {
				w = ((w >> 7) + 1) << 7;
			}
			h = image.h;
			if (h & 0x7f) {
				h = ((h >> 7) + 1) << 7;
			}
			printf("image file size v. actual size: %d x %d v. %d x %d\n", image.w, image.h, w, h);
			image.upscale(w, h);
		}

		input.alloc(w * h);
		float min = +INFINITY;
		float max = -INFINITY;
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int i = y * w + x;
				input.host[i] = image.data[i];
			}
		}

		for (int i = 0; i < w * h; i++) {
			min = std::min(min, input.host[i].x);
			min = std::min(min, input.host[i].y);
			min = std::min(min, input.host[i].z);
			max = std::max(max, input.host[i].x);
			max = std::max(max, input.host[i].y);
			max = std::max(max, input.host[i].z);
		}
		input.push();
		return true;
	}
	void saveBitmap(CudaArray<float3>& input, int w, int h, const char* bitmapName, bool asSRGB = false) {
		input.pull();
		std::vector<uint8_t> pixels(w * h * 3);
		if (asSRGB) {
			for (int i = 0; i < w * h; i++) {
				float b = to_standard(input.host[i].z) * 255.0;
				float g = to_standard(input.host[i].y) * 255.0;
				float r = to_standard(input.host[i].x) * 255.0;
				CLAMP(b, 0.0, 255.0);
				CLAMP(g, 0.0, 255.0);
				CLAMP(r, 0.0, 255.0);
				pixels[i * 3 + 0] = uint8_t(b);
				pixels[i * 3 + 1] = uint8_t(g);
				pixels[i * 3 + 2] = uint8_t(r);
			}
		}
		else {
			for (int i = 0; i < w * h; i++) {
				float b = input.host[i].z * 255.0;
				float g = input.host[i].y * 255.0;
				float r = input.host[i].x * 255.0;
				CLAMP(b, 0.0, 255.0);
				CLAMP(g, 0.0, 255.0);
				CLAMP(r, 0.0, 255.0);
				pixels[i * 3 + 0] = uint8_t(b);
				pixels[i * 3 + 1] = uint8_t(g);
				pixels[i * 3 + 2] = uint8_t(r);
			}
		}
		exportBMP(pixels, w, h, bitmapName);
	}
}

class Main {
	int inputWidth, inputHeight;
	CudaArray<float3> input, bloom, output;
	std::array<CudaArray<float>, 8> downsamples;
	std::array<CudaArray<float3>, 4> mips;
	int numDownsamples = 0;
	int lumIndex = 0;
	float lumAvg = 0.0f;
	cudaStream_t streamLum, streamBloom;
	CudaEventPair globalEventPair, lumCalcEventPair, mipBloomEventPair;

	float EV = -2.0f;
	float grayKey = 0.18f;
	float inScatterAttenuation = 1.0f;
	float4 inScatterWeights = { 8.0f, 4.0f, 2.0f, 1.0f };
public:
	void parseParameterList() {
		FILE* fp = fopen("params.txt", "r");
		const char* params[] = {
			"ExposureValue",
			"GrayKey",
			"InScatterAttenuation",
			"InScatterWeights",
		};

		const int numParams = sizeof(params) / sizeof(const char*);
		auto match = [&](const char* field) 
			{
				for (int i = 0; i < numParams; i++)
					if (0 == strcmp(params[i], field))
						return i;
				return -1;
			};

		if (fp) {
			char line[256];
			while (fgets(line, sizeof(line), fp)) {
				if (line[0] == '#')
					continue;
				char* toks[2];
				toks[0] = strtok(line, " ");
				int p = match(toks[0]);
				if (-1 == p)
					continue;
				toks[1] = strtok(nullptr, " \n");
				switch (p) {
				case 0: 
					EV = atof(toks[1]);
					break;
				case 1:
					grayKey = atof(toks[1]);
					break;
				case 2:
					inScatterAttenuation = atof(toks[1]);
					break;
				case 3:
					char* toktok;
					toktok = strtok(toks[1], ",");
					if (toktok)
						inScatterWeights.x = atof(toktok);
					toktok = strtok(nullptr, ",");
					if (toktok)
						inScatterWeights.y = atof(toktok);
					toktok = strtok(nullptr, ",");
					if (toktok)
						inScatterWeights.z = atof(toktok);
					toktok = strtok(nullptr, " ,\n");
					if (toktok)
						inScatterWeights.w = atof(toktok);
					break;
				default: break;
				}
			}
			fclose(fp);
		}
	}

	bool init() {
		auto createDownsample = [](CudaArray<float>& output, int& size)
			{
				output.free();
				if (size >= WARP_SIZE) {
					size /= WARP_SIZE;
					output.alloc(size);
					return true;
				}
				return false;
			};
		auto createMip = [](CudaArray<float3>& output, int& w, int& h)
			{
				output.free();
				if (w >= 2 && h >= 2) {
					w >>= 1;
					h >>= 1;
					output.alloc(w * h);
					return true;
				}
				return false;
			};

		parseParameterList();
		printf("EV = %.2f, GrayKey = %.2f, InScatterAttenuation = %.2f, InScatterWeights = {%.2f,%.2f,%.2f,%.2f}\n",
			EV, grayKey, inScatterAttenuation, inScatterWeights.x, inScatterWeights.y, inScatterWeights.z, inScatterWeights.w);

		if (!globals::loadHDR(input, inputWidth, inputHeight, "input4.hdr"))
			return false;

		cudaStreamCreate(&streamLum);
		cudaStreamCreate(&streamBloom);

		globalEventPair.init();
		lumCalcEventPair.init();
		mipBloomEventPair.init();

		int size = inputWidth * inputHeight;
		bloom.alloc(size);
		output.alloc(size);
		while (createDownsample(downsamples[numDownsamples], size)) {
			numDownsamples++;
		}
		int w = inputWidth;
		int h = inputHeight;
		createMip(mips[0], w, h);
		createMip(mips[1], w, h);
		createMip(mips[2], w, h);
		createMip(mips[3], w, h);

		cudaDeviceSynchronize();
		return true;
	}

	void calculateLuminosity() {
		dim3 block = dim3(WARP_SIZE, 1, 1);
		dim3 grid = dim3(input.count() / WARP_SIZE, 1, 1);
		lumIndex = 0;

		lumCalcEventPair.setStream(streamLum);
		lumCalcEventPair.start();
		log_average_kernel << <grid, block, 0, streamLum >> > (input.devptr, downsamples[lumIndex].devptr);
		while (downsamples[lumIndex].count() >= WARP_SIZE) {
			//downsamples[lumIndex].pull();
			//printf("size @ %d: %d (value %f)\n", index, downsamples[lumIndex].count(), to_srgb(expf(downsamples[lumIndex].host[0])));
			grid = dim3(downsamples[lumIndex + 1].count(), 1, 1);
			average_kernel << <grid, block, 0, streamLum >> > (downsamples[lumIndex].devptr, downsamples[lumIndex + 1].devptr);
			++lumIndex;
		}
		lumCalcEventPair.finish();
	}

	void setSymbols() {
		setCudaExposure(EV);
		setCudaGrayKey(grayKey);
		setCudaBloomAlpha(inScatterAttenuation);
		setCudaBloomFactor(inScatterWeights);
	}

	void buildBlurMips() {
		int w = inputWidth;
		int h = inputHeight;

		dim3 block = dim3(BLOCK_DIM, BLOCK_DIM);
		dim3 grid = dim3(w / BLOCK_DIM, h / BLOCK_DIM);

		mipBloomEventPair.setStream(streamBloom);
		mipBloomEventPair.start();
		bloom_extract_and_blur_5x5_kernel << <grid, block, 0, streamBloom >> > (input.devptr, output.devptr, w, h);
		grid.x /= 2;
		grid.y /= 2;
		//mip_kernel << <grid, block, 0, stream >> > (output.devptr, w, h, mips[0].devptr);
		mip_and_blur_5x5_kernel << <grid, block, 0, streamBloom >> > (output.devptr, w, h, mips[0].devptr);
		w /= 2;
		h /= 2;
		grid.x /= 2;
		grid.y /= 2;
		mip_and_blur_5x5_kernel << <grid, block, 0, streamBloom >> > (mips[0].devptr, w, h, mips[1].devptr);
		w /= 2;
		h /= 2;
		grid.x /= 2;
		grid.y /= 2;
		mip_and_blur_5x5_kernel << <grid, block, 0, streamBloom >> > (mips[1].devptr, w, h, mips[2].devptr);
		w /= 2;
		h /= 2;
		grid.x /= 2;
		grid.y /= 2;
		mip_and_blur_5x5_kernel << <grid, block, 0, streamBloom >> > (mips[2].devptr, w, h, mips[3].devptr);

		w = inputWidth;
		h = inputHeight;
		grid = dim3(w / BLOCK_DIM, h / BLOCK_DIM);
		combine_mips_and_blur_5x5_kernel << <grid, block, 0, streamBloom >> > (bloom.devptr, w, h, mips[0].devptr, mips[1].devptr, mips[2].devptr, mips[3].devptr);

		mipBloomEventPair.finish();
	}

	void compose() {
		lumAvg = fmaxf(lumAvg, 1e-4f);
		dim3 block = dim3(BLOCK_DIM, BLOCK_DIM);
		dim3 grid = dim3(inputWidth / BLOCK_DIM, inputHeight / BLOCK_DIM);
		globalEventPair.start();
		//bloom_extract_kernel << <grid, block >> > (input.devptr, bloom.devptr, inputWidth, inputHeight);
		bloom_apply_kernel << <grid, block >> > (input.devptr, bloom.devptr, output.devptr, inputWidth, inputHeight, lumAvg);
		globalEventPair.finish();
		globalEventPair.sync();
	}

	void run() {
		setSymbols();

		globalEventPair.start();
		buildBlurMips();
		calculateLuminosity();
		mipBloomEventPair.sync();
		lumCalcEventPair.sync();
		cudaStreamSynchronize(streamBloom);
		cudaStreamSynchronize(streamLum);
		cudaDeviceSynchronize();
		globalEventPair.finish();
		globalEventPair.sync();

		printf("total time taken to calculate luminosity and create blurred mips: %f ms\n", globalEventPair.elapsedMs());
		printf(" + time taken to calculate luminosity...: %f ms\n", lumCalcEventPair.elapsedMs());
		printf(" + time taken to create blurred mipmaps.: %f ms\n", mipBloomEventPair.elapsedMs());
		globals::saveBitmap(bloom, inputWidth, inputHeight, "linear_bloom.bmp");
		globals::saveBitmap(bloom, inputWidth, inputHeight, "bloom.bmp", true);

		downsamples[lumIndex].pull();
		for (int i = 0; i < downsamples[lumIndex].count(); i++)
			lumAvg += downsamples[lumIndex].host[i];
		lumAvg /= float(downsamples[lumIndex].count());
		lumAvg = expf(lumAvg);

		printf("AVERAGE LUMINANCE: %f, (standard %f)\n", lumAvg, to_standard(lumAvg));

		globals::saveBitmap(mips[0], inputWidth / 2, inputHeight / 2, "mip0.bmp");
		globals::saveBitmap(mips[1], inputWidth / 4, inputHeight / 4, "mip1.bmp");
		globals::saveBitmap(mips[2], inputWidth / 8, inputHeight / 8, "mip2.bmp");
		globals::saveBitmap(mips[3], inputWidth / 16, inputHeight / 16, "mip3.bmp");

		cudaDeviceSynchronize();
		cudaStreamSynchronize(0);

		compose();
		printf("total time taken to compose tone-mapped output: %f ms\n", globalEventPair.elapsedMs());
		globals::saveBitmap(output, inputWidth, inputHeight, "tone_mapped.bmp");
	}


	void term() {
		input.free();
		bloom.free();
		output.free();
		for (int i = 0; i < numDownsamples; i++)
			downsamples[i].free();
		for (int i = 0; i < mips.size(); i++)
			mips[i].free();

		globalEventPair.destroy();
		mipBloomEventPair.destroy();
		lumCalcEventPair.destroy();

		cudaStreamDestroy(streamLum);
		cudaStreamDestroy(streamBloom);
	}
};



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

	Main main;
	if (main.init()) {
		main.run();
		main.term();
	}

	status = cudaDeviceReset();
	if (status != cudaSuccess) {
		printf("cudaDeviceReset failed: %s\n", cudaGetErrorString(status));
	}

	printf("goodbye!\n");
	return 0;
}
