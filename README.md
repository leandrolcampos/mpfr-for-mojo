> [!NOTE]
> This project is under development. For more information, see the [Roadmap](#roadmap) section.

# MPFR for Mojo 🔥

This project is a proof of concept (PoC) that shows how we can use the [GNU MPFR](https://www.mpfr.org/) library as a "gold standard" to test the correctness of mathematical functions implemented in [Mojo](https://www.modular.com/mojo).

MPFR is an efficient C library for multiple-precision floating-point computations with _correct rounding_. It is used to test numerical routines in projects such as [CORE-MATH](https://core-math.gitlabpages.inria.fr/), [LLVM-libc](https://libc.llvm.org/index.html), and [RLIBM](https://people.cs.rutgers.edu/~sn349/rlibm/).

By comparing the outputs of our custom Mojo functions with MPFR, we can ensure our implementations are correctly rounded or, at least, as accurate as possible within the memory or latency requirements.

With just a few lines of code, we can test the correctness of a numerical routine under all rounding modes required by the IEEE 754 standard:

```mojo
fn test_sqrt_bf16() raises:
    alias ROUNDING_MODES = available_rounding_modes()

    @parameter
    for i in range(len(ROUNDING_MODES)):
        alias ROUNDING_MODE = ROUNDING_MODES[i]

        var sqrt_checker = UnaryOperatorChecker[
            DType.bfloat16,
            mpfr.sqrt,
            math.sqrt,
            rounding_mode=ROUNDING_MODE,
            default_ulp_tolerance=0.5,
        ]()

        sqrt_checker.assert_special_values()
        sqrt_checker.assert_negative_normals[count=101]()
        sqrt_checker.assert_negative_subnormals[count=11]()
        sqrt_checker.assert_positive_subnormals[count=101]()
        sqrt_checker.assert_positive_normals[count=1_001]()
```

In the above example, we test the implementation of the `sqrt` function available in the Mojo standard library. We expect the specialization of this function for `bfloat16` is correctly rounded or, if not, that the approximation error is at most 0.5 ULP.

## Table of Contents

- [Getting Started](#getting-started)
- [Running](#running)
- [Exploring the Project](#exploring-the-project)
  * [Lower-Precision Floating-Point Types](#lower-precision-floating-point-types)
  * [Rounding Modes](#rounding-modes)
  * [Error Measure](#error-measure)
  * [Roadmap](#roadmap)
- [Getting the Best Efficiency Out of MPFR](#getting-the-best-efficiency-out-of-mpfr)
- [License](#license)
- [References](#references)

## Getting Started

First install [Magic](https://docs.modular.com/magic/#install-magic), a package manager and virtual environment manager for Mojo and other languages.

Then clone this repository:

```bash
git clone https://github.com/leandrolcampos/mpfr-for-mojo.git
```

## Running

To execute smoke and unit tests with the GNU MPFR library, run the following Magic commands.

- For the _round-to-nearest-ties-to-even_ mode, the default rounding mode in the IEEE 754 standard:

    ```
    magic run test
    ```

- For all four mandated rounding modes in the IEEE 754 standard:

    ```
    magic run test-all
    ```

> [!IMPORTANT]
> If a given floating-point type does not support a specific rounding mode in the underlying target, the respective test is skipped.

## Exploring the Project

This section highlights key components of the PoC, including how we handle lower-precision floating-point types and how we manage rounding modes in Mojo.

### Lower-Precision Floating-Point Types

Out of the box, we can set an MPFR value from or convert it to a `float32` or `float64` value. But in Mojo, which is being designed mainly for AI workloads, other floating-point types can be equally or even more important, depending on the scenario.

That's the reason this PoC demonstrates how we can extend the GNU MPFR library to work with `float16` and `bfloat16`.

Setting an MPFR value from a `float16` or `bfloat16` is trivial. In fact, any value in these floating-point types is representable in `float32`, which allows us to promote it to the latter type without data loss and then call `mpfr_set_flt`.

But converting an MPFR value to a narrower type through the `float32` pathway can result in a **double rounding error**, as you can see in the code snippet below.

```mojo
fn double_rounding_error():
    var x = MpfrFloat[DType.bfloat16]("1.0039063")
    # This value is slightly above the midpoint between 1.0 and the next
    # representable bfloat16 value, 1.0078125. The midpoint is 1.00390625.

    var x_bf16: BFloat16 = x[]
    # In the round-to-nearest-ties-to-even mode, `x` casted to bfloat16 rounds
    # up to 1.0078125, since `x` is above that midpoint.

    var x_fp32_bf16 = BFloat16(mpfr.get_flt(x))
    # But in the same rounding mode, `x` casted to float32 rounds down exactly
    # to the midpoint, because `x` is too close to that number - less than half
    # of the distance between the midpoint and the next representable value in
    # float32. By "ties to even", `BFloat16(1.00390625)` rounds down to 1.0.

    print("The value `x` correctly rounded to bfloat16:", x_bf16)  # 1.0078125
    print("The value `x` double-rounded to bfloat16:", x_fp32_bf16)  # 1.0
```

In the example above, note that the expression `val[]` calls the `__getitem__` dunder method of our `MpfrFloat` object. Under the hood, this method uses a custom, generic conversion pipeline implemented within this project that avoids the double rounding error.

### Rounding Modes

Mathematical function implementations in projects such as CORE-MATH, LLVM-libc, and RLIBM are designed to be correctly rounded for all rounding modes required by the IEEE 754 standard.

To test the correctness of numerical routines for different rounding modes, this project uses LLVM intrinsics to manipulate the floating-point environment. The current rounding mode of the floating-point environment is then used to perform computations and produce the desired outputs, which are subsequently compared with the MPFR results.

The following code snippet is a simple example of how we could implement a unit test for different rounding modes.

```mojo
fn test_sqrt_simple() raises:
    alias FLOAT_TYPE = DType.float32
    alias ROUNDING_MODE = RoundingMode.UPWARD

    var expected = MpfrFloat[FLOAT_TYPE, ROUNDING_MODE](2.0)
    _ = mpfr.sqrt(expected, expected)

    with RoundingContext(ROUNDING_MODE):
        if quick_get_rounding_mode[FLOAT_TYPE]() == ROUNDING_MODE:
            var actual = math.sqrt(Scalar[FLOAT_TYPE](2.0))
            assert_equal(expected[], actual)
```

The `RoundingContext` struct is responsible for temporarily changing the floating-point environment. When we enter the context, it sets the new rounding mode. Once we exit it, it automatically restores the previous one.

Meanwhile, the `quick_get_rounding_mode` function infers the effective rounding mode by performing volatile loading and simple floating-point arithmetic operations. This lightweight approach doesn’t rely on directly reading the current rounding mode from the floating-point environment. It can be particularly helpful for floating-point types, such as `bfloat16`, which are not part of the IEEE 754 standard and therefore may not honor the floating-point environment settings.

### Error Measure

In this PoC, we measure errors in terms of _units in the last place (ulp)_, a metric that denotes the magnitude of the last significand digit of a value in the target floating-point format and is widely used for expressing errors of atomic functions such as arithmetic operations, elementary functions, and inner products [[3](#muller2018)]. We adopt the Goldberg definition of $\text{ulp}(x)$ extended to all reals $x$, which states that if $|x| \in [\beta^{e}, \beta^{e+1})$ for some integer $e$, then

$$
\text{ulp}(x) = \beta^{\text{max}(e, e_{\text{min}}) - p + 1} ,
$$

where $\beta$ is the radix (often 2), $p$ is the precision, and $e_{\text{min}}$ is the minimum exponent of the target format. The choice of this definition is due to its popularity [[3](#muller2018)]. In addition, under this definition, rounding to nearest implies a maximal error of 0.5 ulp [[4](#muller2016)]. But notice that the converse is not necessarily true. More precisely, we have [[2](#gladman2024)][[4](#muller2016)]
- For any radix $\beta$, if $X$ is a floating-point number in the target format, then $X = \text{RN}(x) \implies |X - x| \le \frac{1}{2} \text{ulp}(x)$, where $\text{RN}(\cdot)$ is a rounding-to-nearest function.
- If $\beta = 2$ then $|X - x| < \frac{1}{2} \text{ulp}(x) \implies X = \text{RN}(x)$.

For each function, assuming $y$ is the value returned by the function and $z$ the exact result (approximated with the GNU MPFR library using a larger precision than the target one), we use the following formula to compute the ulp error [[1](#brisebarre2024)]:

$$
\text{error}_{\text{ulp}}(y, z) = \frac{|y - z|}{\text{ulp}(z)}.
$$

### Roadmap

Below is our current roadmap, detailing completed tasks and upcoming improvements. By outlining these steps, we aim to give a clear picture of the project’s trajectory and invite feedback from the community.

- [x] Implement a wrapper for the GNU MPFR library.
- [x] Add MPFR as a dependency only in the environment used for testing tasks.
- [x] Add support for correctness testing under different rounding modes.
- [x] Implement a pipeline that converts MPFR values to lower-precision floating-point types avoiding double rounding errors.
- [x] Add a very simple math package just to demonstrate how we can test its functions.
- [x] Add a testing module with routines to compare outputs of math function implementations against MPFR, measuring error in ULP.
- [ ] Add thread-safe routines for exhaustive tests (and, optionally, for checking known hard-to-round cases) to the testing module.
- [ ] Add support for new platforms: `linux-aarch64` and `osx-arm64`.
- [ ] Compile lessons learned and recommendations for math library developers in Mojo, as well as for Mojo language and standard library maintainers.

## Getting the Best Efficiency Out of MPFR

To get the best efficiency out of MPFR, we take into consideration and sometimes enforce performance guidelines from the official library documentation:

- Reuse variables whenever possible;
- Allocate or clear variables outside of loops;
- Pass temporary variables to subroutines instead of allocating them inside the subroutines;
- Do not perform unnecessary copies;
- Avoid auxiliary variables: for example, to compute `a = a + b`, use `mpfr_add(a, a, b, ...)`.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## References

[<a id="brisebarre2024">1</a>] Nicolas Brisebarre, Guillaume Hanrot, Jean-Michel Muller, Paul Zimmermann. 2024. Correctly-rounded evaluation of a function: why, how, and at what cost? [[Link](https://hal.science/hal-04474530v1)]

[<a id="gladman2024">2</a>] Brian Gladman, Vincenzo Innocente, John Mather, Paul Zimmermann. 2024. Accuracy of mathematical functions in single, double, double extended, and quadruple precision. [[Link](https://inria.hal.science/hal-03141101v7)]

[<a id="muller2018">3</a>] Jean-Michel Muller, Nicolas Brunie, Florent de Dinechin, Claude-Pierre Jeannerod, Mioara Joldes, Vincent Lefèvre, Guillaume Melquiond, Nathalie Revol, and Serge Torres. 2018. Handbook of floating-point arithmetic. Springer International Publishing. [[Link](https://doi.org/10.1007/978-3-319-76526-6)]

[<a id="muller2016">4</a>] Jean-Michel Muller. 2016. Elementary functions. Birkhäuser Boston. [[Link](https://doi.org/10.1007/978-1-4899-7983-4)]
