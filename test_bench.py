import torch
import os

# Set PYTORCH_CUDA_ALLOC_CONF environment variable
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

# Explanation: By setting PYTORCH_CUDA_ALLOC_CONF to "caching_allocator",
# we enable the caching memory allocator, which improves memory management efficiency.

# Create a CUDA tensor
x = torch.randn(1000, 1000).cuda()

# Explanation: Here, we create a CUDA tensor using the torch.randn() function.
# Since PYTORCH_CUDA_ALLOC_CONF is set, the tensor will be allocated using the caching allocator.

# Perform some computations
y = x + x.t()
z = torch.matmul(y, y)

# Explanation: We perform some computations on the CUDA tensor.
# The caching allocator manages the memory allocation and reuse efficiently,
# reducing the overhead of memory allocation and deallocation operations.

# Clear memory explicitly (optional)
del x, y, z

# Explanation: Clearing the variables is optional, but it can help release GPU memory
# before subsequent operations to avoid excessive memory usage.

# Reset PYTORCH_CUDA_ALLOC_CONF environment variable (optional)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

# Explanation: Resetting PYTORCH_CUDA_ALLOC_CONF to an empty string restores
# the default memory allocator behavior in PyTorch.

# Continue with other operationsmport torch
from torch.utils import benchmark

#typ = torch.float16
#n = 1024 * 16
#a = torch.randn(n, n).type(typ).cuda()
#b = torch.randn(n, n).type(typ).cuda()
#
#t = benchmark.Timer(
#      stmt='a @ b',
#      globals={'a': a, 'b': b})
#
#x = t.timeit(50)
#print('float16 ,',2*n**3 / x.median /1e12)
#
#
typ = torch.float32
n = 1024 * 16
a = torch.randn(n, n).type(typ).cuda()
b = torch.randn(n, n).type(typ).cuda()

t = benchmark.Timer(
              stmt='a @ b',
                    globals={'a': a, 'b': b})

x = t.timeit(50)
print('float32 ,',2*n**3 / x.median /1e12)


#typ = torch.float64
#n = 1024 * 16
#a = torch.randn(n, n).type(typ).cuda()
#b = torch.randn(n, n).type(typ).cuda()
#
#t = benchmark.Timer(
#              stmt='a @ b',
#                    globals={'a': a, 'b': b})
#
#x = t.timeit(50)
#print('float64 ,',2*n**3 / x.median /1e12)

