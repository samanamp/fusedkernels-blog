---
title: "Sample Post: LaTeX and Code Blocks"
pubDatetime: 2025-09-06T12:00:00Z
modDatetime: 2025-09-06T12:00:00Z
author: "The Fused Kernels Team"
draft: false
tags:
  - example
  - tech
description: "A sample post demonstrating LaTeX equations and code blocks in Astro."
---

This is a sample blog post to demonstrate the use of LaTeX and code blocks within your articles. This theme uses KaTeX to render mathematical notations.

## LaTeX Equations

You can write inline equations like $E = mc^2$ by wrapping them in single dollar signs.

For more complex, block-level equations, you can use double dollar signs:

$$
\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$

This is the quadratic formula, a staple in algebra.

## Code Blocks

Syntax highlighting is handled by Shikiji, and it supports a wide variety of languages. Here is an example of a Python code block:

```python frame="terminal" title="src/my-test-file.js"
def fibonacci(n):
    """Generate the Fibonacci sequence up to n."""
    a, b = 0, 1
    while a < n: // [!code highlight]
        print(a, end=' ')
        a, b = b, a+b // [!hl]
    print()

fibonacci(100)
```

look at this:

```bash withOutput
> pwd

/usr/home/boba-tan/programming
```
dd


```sh frame="none"
echo "Look ma, no frame!"
```
and this

```ps frame="code" title="PowerShell Profile.ps1"
# Without overriding, this would be a terminal frame
function Watch-Tail { Get-Content -Tail 20 -Wait $args }
New-Alias tail Watch-Tail
```

And that's it! You can now easily combine mathematical notations and code in your blog posts.
