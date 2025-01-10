# MPFR for Mojo 🔥

This project is a proof of concept (PoC) that demonstrates how we can use the [MPFR](https://www.mpfr.org/) library as a "gold standard" to test the correctness of mathematical functions implemented in Mojo.

MPFR is an efficient C library for multiple-precision floating-point computations with _correct rounding_. It is used to test numerical routines in projects such as [CORE-MATH](https://core-math.gitlabpages.inria.fr/) and [LLVM-libc](https://libc.llvm.org/index.html).

By comparing the outputs of our custom Mojo functions with MPFR, we can ensure our implementations are correctly rounded or, at least, as accurate as possible within requirements on code size, memory footprint or latency.

## Getting the Best Efficiency Out of MPFR

To get the best efficiency out of MPFR, we take into consideration and sometimes enforce performance guidelines from the official library documentation:

- Reuse variables whenever possible;
- Allocate or clear variables outside of loops;
- Pass temporary variables to subroutines instead of allocating them inside the subroutines;
- Do not perform unnecessary copies;
- Avoid auxiliary variables: for example, to compute `a = a + b`, use `mpfr_add(a, a, b, ...)`.

## License

This project is licensed under the [Apache License 2.0](LICENSE). 
