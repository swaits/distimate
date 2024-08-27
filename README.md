# distimate

`distimate` is a Rust crate that provides probability distributions specifically
designed for estimation and risk analysis scenarios. It offers implementations
of various distributions that are commonly used in project management, cost
estimation, and other fields where uncertainty needs to be modeled.

## Features

- Implementations of PERT, Triangular, and Normal distributions tailored for
  estimation purposes
- Consistent API across all distributions
- Support for common statistical operations (mean, variance, skewness, etc.)
- Sampling capabilities for Monte Carlo simulations
- Range-respecting implementations that work with min, likely (mode), and max estimates

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
distimate = "<desired version>"
```

## Usage

Here's a quick example of how to use the PERT distribution:

```rust
use distimate::prelude::*;
use distimate::Pert;

fn main() {
    // Create a PERT distribution for a task estimated to take
    // between 1 and 3 days, most likely 2 days
    let task_duration = Pert::new(1.0, 2.0, 3.0).unwrap();

    println!("Expected duration: {:.2} days", task_duration.expected_value());
    println!("Optimistic estimate (P5): {:.2} days", task_duration.optimistic_estimate());
    println!("Pessimistic estimate (P95): {:.2} days", task_duration.pessimistic_estimate());
    println!("Probability of completing within 2.5 days: {:.2}%",
             task_duration.probability_of_completion(2.5) * 100.0);
    println!("Risk of exceeding 2.5 days: {:.2}%",
             task_duration.risk_of_overrun(2.5) * 100.0);

    let (lower, upper) = task_duration.confidence_interval(0.9);
    println!("90% Confidence Interval: {:.2} to {:.2} days", lower, upper);
}
```

## Distributions

### PERT Distribution

The PERT (Program Evaluation and Review Technique) distribution is commonly used
in project management and risk analysis. It's defined by minimum, most likely
(mode), and maximum values.

```rust
let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
```

### Triangular Distribution

The Triangular distribution is often used when limited sample data is available.
It's also defined by minimum, most likely (mode), and maximum values.

```rust
let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
```

### Normal Distribution

The Normal (Gaussian) distribution is implemented with modifications to respect
a given range. It's suitable for natural phenomena and when values are expected
to be symmetrically distributed around a mean.

```rust
let normal = Normal::new(1.0, 2.0, 3.0).unwrap();
```

## Common Operations

All distributions implement common traits for statistical operations:

- `Distribution`: Provides `mean()`, `variance()`, `skewness()`, and `entropy()`
- `Median`: Provides `median()`
- `Mode`: Provides `mode()`
- `Continuous`: Provides `pdf()` and `ln_pdf()`
- `ContinuousCDF`: Provides `cdf()` and `inverse_cdf()`
- `Min` and `Max`: Provide `min()` and `max()`

Additionally, all distributions can be sampled using the `sample()` method,
which is useful for Monte Carlo simulations.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.
