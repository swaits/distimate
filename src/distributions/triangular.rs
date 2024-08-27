//! Triangular distribution implementation.
//!
//! This module provides an implementation of the Triangular distribution, which is often used
//! in project management, risk analysis, and other fields where limited sample data is available.
//! The Triangular distribution is defined by its minimum, most likely (mode), and maximum values.
//!
//! The Triangular distribution is particularly useful when the relationship between variables is known,
//! but data is scarce. It provides a simple and intuitive way to model situations where you can estimate
//! the minimum, maximum, and most likely values.
//!
//! # Examples
//!
//! ```
//! use distimate::prelude::*;
//! use distimate::Triangular;
//! use approx::assert_relative_eq;
//!
//! let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
//! assert_eq!(triangular.mode(), 2.0);
//! assert_relative_eq!(triangular.mean().unwrap(), 2.0, epsilon = 1e-6);
//! ```

use crate::prelude::*;
use statrs::distribution::Triangular as StatrsTriangular;

/// Represents a Triangular distribution.
///
/// The Triangular distribution is defined by its minimum, most likely (mode), and maximum values.
///
/// # Examples
///
/// ```
/// use distimate::prelude::*;
/// use distimate::Triangular;
/// use approx::assert_relative_eq;
///
/// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
/// assert_eq!(triangular.min(), 1.0);
/// assert_eq!(triangular.mode(), 2.0);
/// assert_eq!(triangular.max(), 3.0);
/// assert_relative_eq!(triangular.mean().unwrap(), 2.0, epsilon = 1e-4);
/// ```
#[derive(Debug, Clone)]
pub struct Triangular {
    inner: StatrsTriangular,
}

impl Triangular {
    /// Creates a new Triangular distribution with the given minimum, likely, and maximum values.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value of the distribution
    /// * `likely` - The most likely value (mode) of the distribution
    /// * `max` - The maximum value of the distribution
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `Triangular` instance if the parameters are valid,
    /// or an `Error` if the parameters are invalid.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `min` is greater than or equal to `max`
    /// - `likely` is less than `min` or greater than `max`
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Triangular;
    ///
    /// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
    /// assert_eq!(triangular.min(), 1.0);
    /// assert_eq!(triangular.mode(), 2.0);
    /// assert_eq!(triangular.max(), 3.0);
    /// ```
    pub fn new(min: f64, likely: f64, max: f64) -> Result<Self> {
        if min >= max {
            return Err(Error::InvalidRange { min, max });
        }
        if likely < min || likely > max {
            return Err(Error::InvalidMode { min, likely, max });
        }

        Ok(Triangular {
            inner: StatrsTriangular::new(min, max, likely)?,
        })
    }
}

/// Implementation of the `EstimationDistribution` trait for the Triangular distribution.
///
/// This trait marks the Triangular distribution as suitable for use in estimation contexts.
/// It doesn't add any new methods, but signals that this distribution is appropriate
/// for modeling uncertain estimates or durations.
///
/// # Examples
///
/// ```
/// use distimate::prelude::*;
/// use distimate::Triangular;
///
/// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
/// // The fact that we can create a Triangular instance demonstrates
/// // that it implements EstimationDistribution
/// ```
impl EstimationDistribution for Triangular {}

/// Implementation of the `Distribution` trait for the Triangular distribution.
///
/// This implementation provides methods to calculate various statistical properties
/// of the Triangular distribution.
impl Distribution<f64> for Triangular {
    /// Calculates the mean of the Triangular distribution.
    ///
    /// # Returns
    ///
    /// The mean of the distribution as an `Option<f64>`. Always returns `Some(value)`
    /// as the Triangular distribution always has a defined mean.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Triangular;
    /// use approx::assert_relative_eq;
    ///
    /// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(triangular.mean().unwrap(), 2.0, epsilon = 1e-6);
    /// ```
    fn mean(&self) -> Option<f64> {
        self.inner.mean()
    }

    /// Calculates the variance of the Triangular distribution.
    ///
    /// # Returns
    ///
    /// The variance of the distribution as an `Option<f64>`. Always returns `Some(value)`
    /// as the Triangular distribution always has a defined variance.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Triangular;
    /// use approx::assert_relative_eq;
    ///
    /// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(triangular.variance().unwrap(), 0.1666667, epsilon = 1e-6);
    /// ```
    fn variance(&self) -> Option<f64> {
        self.inner.variance()
    }

    /// Calculates the skewness of the Triangular distribution.
    ///
    /// # Returns
    ///
    /// The skewness of the distribution as an `Option<f64>`. Always returns `Some(value)`
    /// as the Triangular distribution always has a defined skewness.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Triangular;
    /// use approx::assert_relative_eq;
    ///
    /// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(triangular.skewness().unwrap(), 0.0, epsilon = 1e-6);
    /// ```
    fn skewness(&self) -> Option<f64> {
        self.inner.skewness()
    }

    /// Calculates the entropy of the Triangular distribution.
    ///
    /// The entropy is a measure of the average amount of information contained in the distribution.
    ///
    /// # Returns
    ///
    /// The entropy of the distribution as an `Option<f64>`. Always returns `Some(value)`
    /// as the Triangular distribution always has a defined entropy.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Triangular;
    /// use approx::assert_relative_eq;
    ///
    /// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(triangular.entropy().unwrap(), 0.5, epsilon = 1e-6);
    /// ```
    fn entropy(&self) -> Option<f64> {
        self.inner.entropy()
    }
}

/// Implementation of the `Median` trait for the Triangular distribution.
impl Median<f64> for Triangular {
    /// Calculates the median of the Triangular distribution.
    ///
    /// The median is the value separating the higher half from the lower half of the distribution.
    ///
    /// # Returns
    ///
    /// The median of the distribution as an f64 value.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Triangular;
    /// use approx::assert_relative_eq;
    ///
    /// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(triangular.median(), 2.0, epsilon = 1e-6);
    /// ```
    fn median(&self) -> f64 {
        self.inner.median()
    }
}

/// Implementation of the `Mode` trait for the Triangular distribution.
impl Mode<f64> for Triangular {
    /// Returns the mode of the Triangular distribution.
    ///
    /// The mode is the most likely value of the distribution, which is one of the
    /// defining parameters of the Triangular distribution.
    ///
    /// # Returns
    ///
    /// The mode of the distribution as an f64 value.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Triangular;
    ///
    /// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
    /// assert_eq!(triangular.mode(), 2.0);
    /// ```
    fn mode(&self) -> f64 {
        self.inner.mode().unwrap()
    }
}

/// Implementation of the `Continuous` trait for the Triangular distribution.
///
/// This implementation provides methods to calculate the probability density function (PDF)
/// and its natural logarithm for the Triangular distribution.
impl Continuous<f64, f64> for Triangular {
    /// Calculates the probability density function (PDF) of the Triangular distribution at a given point.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to calculate the PDF
    ///
    /// # Returns
    ///
    /// The value of the PDF at the given point as an f64.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Triangular;
    /// use approx::assert_relative_eq;
    ///
    /// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(triangular.pdf(2.0), 1.0, epsilon = 1e-6);
    /// assert_eq!(triangular.pdf(0.0), 0.0); // Outside the distribution's range
    /// ```
    fn pdf(&self, x: f64) -> f64 {
        self.inner.pdf(x)
    }

    /// Calculates the natural logarithm of the probability density function (PDF) of the Triangular distribution at a given point.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to calculate the log PDF
    ///
    /// # Returns
    ///
    /// The natural logarithm of the PDF at the given point as an f64.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Triangular;
    /// use approx::assert_relative_eq;
    ///
    /// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(triangular.ln_pdf(2.0), 0.0, epsilon = 1e-6);
    /// assert_eq!(triangular.ln_pdf(0.0), f64::NEG_INFINITY); // Outside the distribution's range
    /// ```
    fn ln_pdf(&self, x: f64) -> f64 {
        self.inner.ln_pdf(x)
    }
}

/// Implementation of the `ContinuousCDF` trait for the Triangular distribution.
impl ContinuousCDF<f64, f64> for Triangular {
    /// Calculates the cumulative distribution function (CDF) of the Triangular distribution at a given point.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to calculate the CDF
    ///
    /// # Returns
    ///
    /// The value of the CDF at the given point as an f64, in the range [0, 1].
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Triangular;
    /// use approx::assert_relative_eq;
    ///
    /// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(triangular.cdf(2.0), 0.5, epsilon = 1e-6);
    /// assert_eq!(triangular.cdf(1.0), 0.0);
    /// assert_eq!(triangular.cdf(3.0), 1.0);
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        self.inner.cdf(x)
    }

    /// Calculates the inverse of the cumulative distribution function (inverse CDF)
    /// of the Triangular distribution for a given probability.
    ///
    /// # Arguments
    ///
    /// * `p` - The probability for which to calculate the inverse CDF, must be in the range [0, 1]
    ///
    /// # Returns
    ///
    /// The value x for which the CDF of the distribution equals the given probability.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Triangular;
    /// use approx::assert_relative_eq;
    ///
    /// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(triangular.inverse_cdf(0.5), 2.0, epsilon = 1e-4);
    /// assert_relative_eq!(triangular.inverse_cdf(0.0), 1.0, epsilon = 1e-4);
    /// assert_relative_eq!(triangular.inverse_cdf(1.0), 3.0, epsilon = 1e-4);
    /// ```
    fn inverse_cdf(&self, p: f64) -> f64 {
        self.inner.inverse_cdf(p)
    }
}

/// Implementation of the `Min` trait for the Triangular distribution.
impl Min<f64> for Triangular {
    /// Returns the minimum value of the Triangular distribution.
    ///
    /// # Returns
    ///
    /// The minimum value of the distribution as an f64.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Triangular;
    ///
    /// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
    /// assert_eq!(triangular.min(), 1.0);
    /// ```
    fn min(&self) -> f64 {
        self.inner.min()
    }
}

/// Implementation of the `Max` trait for the Triangular distribution.
impl Max<f64> for Triangular {
    /// Returns the maximum value of the Triangular distribution.
    ///
    /// # Returns
    ///
    /// The maximum value of the distribution as an f64.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Triangular;
    ///
    /// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
    /// assert_eq!(triangular.max(), 3.0);
    /// ```
    fn max(&self) -> f64 {
        self.inner.max()
    }
}

/// Implementation of the `rand::distributions::Distribution` trait for the Triangular distribution.
///
/// This implementation allows for random sampling from the Triangular distribution using
/// any random number generator that implements the `rand::Rng` trait.
impl RandDistribution<f64> for Triangular {
    /// Generates a random sample from the Triangular distribution.
    ///
    /// This method uses the inverse transform sampling technique: it generates a uniform
    /// random number between 0 and 1, then applies the inverse CDF to this number to
    /// produce a sample from the Triangular distribution.
    ///
    /// # Arguments
    ///
    /// * `rng` - A random number generator that implements the `rand::Rng` trait
    ///
    /// # Returns
    ///
    /// A random sample from the Triangular distribution as an f64.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Triangular;
    /// use rand::prelude::*;
    ///
    /// let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
    /// let mut rng = StdRng::seed_from_u64(42);  // For reproducibility
    ///
    /// let sample = triangular.sample(&mut rng);
    /// assert!(sample >= 1.0 && sample <= 3.0);
    /// ```
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.inner.sample(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::distributions::Distribution as RandDistribution;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use statrs::statistics::{Data, Distribution, OrderStatistics};

    #[test]
    fn test_new() {
        let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
        assert_eq!(triangular.min(), 1.0);
        assert_eq!(triangular.mode(), 2.0);
        assert_eq!(triangular.max(), 3.0);
    }

    #[test]
    fn test_min() {
        let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
        assert_eq!(triangular.min(), 1.0);
    }

    #[test]
    fn test_max() {
        let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
        assert_eq!(triangular.max(), 3.0);
    }

    #[test]
    fn test_mean() {
        let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
        assert_relative_eq!(triangular.mean().unwrap(), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_median() {
        let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
        assert_relative_eq!(triangular.median(), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mode() {
        let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
        assert_eq!(triangular.mode(), 2.0);
    }

    #[test]
    fn test_variance() {
        let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
        assert_relative_eq!(triangular.variance().unwrap(), 0.1666667, epsilon = 1e-6);
    }

    #[test]
    fn test_skewness() {
        let triangular = Triangular::new(1.0, 2.0, 3.0).unwrap();
        assert_relative_eq!(triangular.skewness().unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_asymmetric_distribution() {
        let triangular = Triangular::new(0.0, 1.0, 5.0).unwrap();
        assert_relative_eq!(triangular.mean().unwrap(), 2.0, epsilon = 1e-6);
        assert_relative_eq!(triangular.median(), 1.83772234, epsilon = 1e-6);
        assert_relative_eq!(triangular.variance().unwrap(), 1.1666667, epsilon = 1e-6);
        assert_relative_eq!(triangular.skewness().unwrap(), 0.47613605, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: InvalidRange { min: 3.0, max: 1.0 }"
    )]
    fn test_invalid_parameters() {
        Triangular::new(3.0, 2.0, 1.0).unwrap(); // min > max
    }

    fn sample_statistics(dist: &Triangular, n: usize) -> (f64, f64, f64, f64) {
        let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
        let samples: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();

        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std_dev = variance.sqrt();
        let mut sorted_samples = samples.clone();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if n % 2 == 0 {
            (sorted_samples[n / 2 - 1] + sorted_samples[n / 2]) / 2.0
        } else {
            sorted_samples[n / 2]
        };

        (mean, variance, std_dev, median)
    }

    #[test]
    fn test_triangular_distribution_statistics() {
        let min = 1.0;
        let mode = 2.0;
        let max = 5.0;
        let dist = Triangular::new(min, mode, max).unwrap();
        let n = 100_000; // Number of samples
        let epsilon = 0.01; // Tolerance for relative equality

        let (sample_mean, sample_variance, sample_std_dev, sample_median) =
            sample_statistics(&dist, n);

        // Test mean
        assert_relative_eq!(
            sample_mean,
            dist.mean().unwrap(),
            epsilon = epsilon,
            max_relative = epsilon
        );

        // Test variance
        assert_relative_eq!(
            sample_variance,
            dist.variance().unwrap(),
            epsilon = epsilon,
            max_relative = epsilon
        );

        // Test standard deviation
        assert_relative_eq!(
            sample_std_dev,
            dist.std_dev().unwrap(),
            epsilon = epsilon,
            max_relative = epsilon
        );

        // Test median
        assert_relative_eq!(
            sample_median,
            dist.median(),
            epsilon = epsilon,
            max_relative = epsilon
        );
    }

    #[test]
    fn test_triangular_distribution_bounds() {
        let min = 1.0;
        let mode = 2.0;
        let max = 3.0;
        let dist = Triangular::new(min, mode, max).unwrap();
        let n = 100_000;
        let mut rng = StdRng::seed_from_u64(42);

        let samples: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();

        assert!(samples.iter().all(|&x| x >= min && x <= max));
        assert!(samples.iter().any(|&x| x < mode));
        assert!(samples.iter().any(|&x| x > mode));
    }

    fn sample_skewness(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let m2 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let m3 = data.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
        m3 / m2.powf(1.5)
    }

    #[test]
    fn test_distribution_statistics() {
        let test_cases = vec![
            (0.0, 5.0, 10.0),       // Standard Triangular
            (0.0, 2.0, 10.0),       // Right-skewed Triangular
            (0.0, 9.0, 10.0),       // Left-skewed Triangular
            (-10.0, 0.0, 10.0),     // Symmetric Triangular with negative values
            (100.0, 400.0, 1000.0), // Triangular with larger values
        ];

        for (min, mode, max) in test_cases {
            let triangular = Triangular::new(min, mode, max).unwrap();
            let n = 25_000; // Number of samples
            let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility

            let samples: Vec<f64> = (0..n).map(|_| triangular.sample(&mut rng)).collect();
            let mut data = Data::new(samples.clone());

            // Mean
            let sample_mean = data.mean().unwrap();
            let theoretical_mean = triangular.mean().unwrap();
            assert_relative_eq!(
                sample_mean,
                theoretical_mean,
                epsilon = 0.1,
                max_relative = 0.1
            );

            // Variance
            let sample_variance = data.variance().unwrap();
            let theoretical_variance = triangular.variance().unwrap();
            assert_relative_eq!(
                sample_variance,
                theoretical_variance,
                epsilon = 0.05,
                max_relative = 0.05
            );

            // Standard deviation
            let sample_std_dev = data.std_dev().unwrap();
            let theoretical_std_dev = triangular.std_dev().unwrap();
            assert_relative_eq!(
                sample_std_dev,
                theoretical_std_dev,
                epsilon = 0.1,
                max_relative = 0.1
            );

            // Skewness
            let sample_skewness = sample_skewness(&samples);
            let theoretical_skewness = triangular.skewness().unwrap();
            assert_relative_eq!(
                sample_skewness,
                theoretical_skewness,
                epsilon = 0.1,
                max_relative = 0.1
            );

            // Median
            let sample_median = data.median();
            let theoretical_median = triangular.median();
            assert_relative_eq!(
                sample_median,
                theoretical_median,
                epsilon = 0.1,
                max_relative = 0.1
            );

            // Mode (we can't reliably estimate the mode from samples, so we'll check if it's within the range)
            assert!(triangular.mode() >= min && triangular.mode() <= max);

            // Min and Max
            let sample_min = data.min();
            let sample_max = data.max();
            assert!(sample_min >= triangular.min());
            assert!(sample_max <= triangular.max());

            // CDF and Inverse CDF
            let percentiles = [0.1, 0.25, 0.5, 0.75, 0.9];
            for p in percentiles.iter() {
                let sample_percentile = data.percentile((100.0 * p) as usize);
                let theoretical_percentile = triangular.inverse_cdf(*p);
                assert_relative_eq!(
                    sample_percentile,
                    theoretical_percentile,
                    epsilon = 0.05,
                    max_relative = 0.05
                );

                let sample_cdf =
                    samples.iter().filter(|&x| x <= &sample_percentile).count() as f64 / n as f64;
                let theoretical_cdf = triangular.cdf(sample_percentile);
                assert_relative_eq!(
                    sample_cdf,
                    theoretical_cdf,
                    epsilon = 0.01,
                    max_relative = 0.01
                );
            }

            println!("Test passed for Triangular({}, {}, {})", min, mode, max);
        }
    }
}
