#define FORRANGE(index, length) for (size_t index = threadIdx.x + blockIdx.x * blockDim.x; index < length; index += gridDim.x * blockDim.x)

extern "C" __global__ void relu_forward_inplace_f(float *x, size_t len)
{
  FORRANGE(i, len) {
    if (x[i] < 0)
      x[i] = 0;
  }
}
