# Calculus Project

This project is part of the Holberton School Machine Learning curriculum and focuses on various calculus operations such as summation, differentiation, integration, and their applications. The directory contains scripts implementing mathematical concepts with detailed explanations.

## Directory Overview

### Summation
1. **`0-sigma_is_for_sum`**
   - Calculates the sum of a sequence:  
     Example: \(\sum_{i=2}^{5} i = 2 + 3 + 4 + 5\)

2. **`1-seegma`**
   - Calculates a summation with a variable:  
     Example: \(\sum_{k=1}^{4} (9i - 2k) = 36i - 20\)

### Product
3. **`2-pi_is_for_product`**
   - Computes the product of a sequence:  
     Example: \(\prod_{i=1}^{m} i = m!\)

4. **`3-pee`**
   - Handles special cases in products:  
     Example: \(\prod_{i=0}^{10} i = 0\)

### Differentiation
5. **`4-hello_derivatives`**
   - Computes the derivative of a polynomial:  
     Example: \(\frac{dy}{dx}\) where \(y = x^4 + 3x^3 - 5x + 1\) results in \(4x^3 + 9x^2 - 5\)

6. **`5-log_on_fire`**
   - Derives logarithmic functions:  
     Example: \(\frac{d}{dx}(x\ln(x)) = \ln(x) + 1\)

7. **`6-voltaire`**
   - Derives logarithmic expressions:  
     Example: \(\frac{d}{dx}(\ln(x^2)) = \frac{2}{x}\)

8. **`7-partial_truths`**
   - Computes partial derivatives:  
     Example: \(\frac{\partial f(x, y)}{\partial y}\) where \(f(x, y) = e^{xy}\) yields \(xe^{xy}\)

9. **`8-all-together`**
   - Computes higher-order partial derivatives:  
     Example: \(\frac{\partial^2}{\partial y \partial x}(e^{x^2y}) = 2x(1 + x^2y)e^{x^2y}\)

### Summation Functions
10. **`9-sum_total.py`**
    - Implements a Python function to compute the sum of squares using the formula:  
      \[
      \text{Sum} = \frac{n(n+1)(2n+1)}{6}
      \]

### Derivative Functions
11. **`10-matisse.py`**
    - Defines a function `poly_derivative` to calculate the derivative of a polynomial represented by a list of coefficients.

### Integration
12. **`11-integral`**
    - Integrates a simple polynomial:  
      Example: \(\int x^3 dx = \frac{x^4}{4} + C\)

13. **`12-integral`**
    - Integrates an exponential function:  
      Example: \(\int e^{2y} dy = \frac{e^{2y}}{2} + C\)

14. **`13-definite`**
    - Computes a definite integral:  
      Example: \(\int_{0}^{3} u^2 du = 9\)

15. **`14-definite`**
    - Handles undefined integrals:  
      Example: \(\int_{-1}^{0} \frac{1}{v} dv = \text{undefined}\)

16. **`15-definite`**
    - Integrates with respect to another variable:  
      Example: \(\int_{0}^{5} x dy = 5x\)

17. **`16-double`**
    - Solves a double integral:  
      Example: \(\int_{1}^{2} \int_{0}^{3} \frac{x^2}{y} dx dy = 9\ln(2)\)

### Integration Functions
18. **`17-integrate.py`**
    - Provides the `poly_integral` function to calculate the integral of a polynomial, returning the coefficients of the resulting polynomial.

---

## Usage

- Each file can be executed directly to demonstrate the mathematical concepts and Python implementations.
- Functions are documented with clear input/output details.

### Python Functions
- `9-sum_total.py`: `summation_i_squared(n)` computes the sum of squares from \(1\) to \(n\).
- `10-matisse.py`: `poly_derivative(poly)` calculates the derivative of a polynomial.
- `17-integrate.py`: `poly_integral(poly, C=0)` calculates the integral of a polynomial.

---

## Requirements
- Python 3.x
- No additional libraries are required.

---

## Author
- Davis

