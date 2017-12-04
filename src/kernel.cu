#define FORRANGE(index, length) for (size_t index = threadIdx.x + blockIdx.x * blockDim.x; index < length; index += gridDim.x * blockDim.x)

extern "C" __global__ void axpy_f(const size_t len, const float alpha, const float *x, float *y)
{
  FORRANGE(i, len)
  {
    y[i] += alpha * x[i];
  }
}
