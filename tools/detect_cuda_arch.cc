#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static void
check(cudaError_t err)
{
    if (err != cudaSuccess) {
        exit(1);
    }
}

static int
min(int x, int y)
{
    return x <= y ? x : y;
}

int main()
{
    int res = INT_MAX;
    cudaDeviceProp prop;
    int count, i, n;

    check(cudaGetDeviceCount(&count));

    for (i = 0; i < count; i++) {
        check(cudaGetDeviceProperties(&prop, i));
        n = prop.major * 10 + prop.minor;
        res = min(res, n);
    }

    printf("%d", res);
    return 0;
}
