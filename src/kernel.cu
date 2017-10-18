#define FORRANGE(index, length) for (size_t index = threadIdx.x + blockIdx.x * blockDim.x; index < length; index += gridDim.x * blockDim.x)

extern "C" __global__ void relu_forward_f(const size_t len, float *x)
{
  FORRANGE(i, len) {
    if (x[i] < 0)
      x[i] = 0;
  }
}

extern "C" __global__ void relu_backward_f(const size_t len, const float *y, float *dy)
{
  FORRANGE(i, len) {
    if (y[i] <= 0)
      dy[i] = 0;
  }
}
