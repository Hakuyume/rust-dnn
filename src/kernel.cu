#define FORRANGE(index, length) for (size_t index = threadIdx.x + blockIdx.x * blockDim.x; index < length; index += gridDim.x * blockDim.x)

extern "C" __global__ void relu_forward_f(float *x, const size_t len)
{
  FORRANGE(i, len) {
    if (x[i] < 0)
      x[i] = 0;
  }
}

extern "C" __global__ void relu_backward_f(const float *y, float *dy, const size_t len)
{
  FORRANGE(i, len) {
    if (y[i] <= 0)
      dy[i] = 0;
  }
}
