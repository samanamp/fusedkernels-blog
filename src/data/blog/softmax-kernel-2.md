---
title: "Fused Softmax::P2::Triton optimization"
pubDatetime: 2025-11-09T12:00:00Z
modDatetime: 2025-11-09T12:00:00Z
author: "Saman"
draft: false
published: true
tags:
  - Benchmarking
description: "Debugging triton kernel optimization issue"
---

As much as I wanted to let it go and jump into Cuda kernel implementation, I couldn't. That nagging voice that says let's see why triton kerenel drops dead.

Now let's get into it. Zooming in, noticed that drops occurs on exact powers of 2.
![softmax-p2-blocksize](../../assets/images/softmax-p2-blocksize.png)

Looking at the code, we can see why it happens:

```python
BLOCK_SIZE=triton.next_power_of_2(n_cols)
```

We're good up to $$ 2^{14} $$ then we have a drop. With in $$ 2^{15} $$ band we see the performance is getting better until we hit $$ 2^{16} $$. That's because early in $$ 2^{15} $$ band we are just utilizing parts of the block, the rest are masked away; lowering the effective bandwidth.

I tried different ways to fix the issue, but wasn't successful. May get back to it after writing the Cuda version.