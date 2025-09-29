---
title: "Vector Addition-part 1"
pubDatetime: 2025-09-30T12:00:00Z
modDatetime: 2025-09-30T12:00:00Z
author: "Saman"
draft: false
tags:
  - Triton kernel
  - tech
description: "Investing vector addition kernel in Triton"
---

Haha, this my first post of this blog. Wanted to bring you on the ride as I'm learning and implementing kernels. It might look like I've figured everything out; but in truth, I'm just few steps ahead figuring out the way.

Anyway, let's get into it. I'll try to follow [triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)'s tutorials, the first one is vector addition. Thec will add my own flavor to it. We're going to implement a Cuda version as well (probably the naive implementation's perf will suck!) to compare to the Triton implementation. And then go in the rabit hole to match the Triton and Pytorch native op performance. How does it sound like? I love it.

Now, let's try to copy and paste the main kernel:
```python
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```
If it's your first time looking at a kernel, definitely you're going to say it's total non-sense. To add to this drama, let me introduce the wrapper:
```python
def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output
```
Almost all of the kernels I've seen have a wrapper. Wrapper mostly acts like interface between user and kernel. Why is that? Because to run kernels correctly, we need to pass in a ton of inputs.

Now where should I start with? the wrapper or the kernel itself? Maybe the wrapper, we go from big picture to details.

The wrapper defines an add(x,y) function. That's exactly like pytorch addition op:
```python
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
```

Now inside the wrapper we have the following line:
```python
output = torch.empty_like(x)
```
where we pre-allocate output memory in the GPU. Some kernels require pre-allocating the output ([Checkout scaled_fp4_quant in vllm](https://github.com/vllm-project/vllm/blob/0307428d65acf5cf1a73a70a7722e076bbb83f22/vllm/_custom_ops.py#L1131)) and some kernels just do it themselves ([checkout moe_wna16_marlin_gemm](kernel-https://github.com/vllm-project/vllm/blob/0307428d65acf5cf1a73a70a7722e076bbb83f22/vllm/_custom_ops.py#L1131)). I haven't found the rule of thumb for this behaviour yet, but will update here when I find it.

Then, let's look at the grid definition part of the wrapper:
```python
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
```
It's time to checkout triton's official page. Quick search gets us to [triton.language.cdiv](https://triton-lang.org/main/python-api/generated/triton.language.cdiv.html). So this function calculates ceiling division of n_elements by Block size. So imagining number of elements are 32 and block size is 1024; we will have just one grid. And this grid is number of programs that triton will launch. Beyond this is magic that Triton handles and probably we will get to it in the 3rd part.

To be honest, I think you should really like kernels to read this far. We're just getting started.

Going to kernel itself, first thing we see is `@triton.jit` decorator. This tells triton to take the python function and compile it into optimized GPU kernel. The compilation is done when we're execution the kernel, hense calling it JIT or Just-In-Time. Will later get into details of this.

Next, we see x, y and output are passed as pointers (e.g. x_ptr) rather than PyTorch Tensors. Triton kernels operate on lower-level memory pointers. Each of these pointers are memory addresses pointing to the beginning of vector data in gpu. This gives us fine grained control over memory operations.

Do you remember we defined the grid size above? We can detect which program in the grid we are executing using this:
```python
pid = tl.program_id(axis=0)
```

I imagine the main code's comments on offsets, block_start and memory safety (using masks) and the rest of the code is pretty good. So skipping it.

Before reading the next part, please load this on a Jupyter notebook and play with it. If you want to poke around the variables, you can print them with 
```python
tl.device_print("pid", pid)
```

Goodluck!