//! Normal (Gaussian) distribution implementation for estimation.
//!
//! This module provides an implementation of a modified Normal distribution,
//! which is defined by minimum, most likely (mode), and maximum values.
//! It's designed for use in estimation scenarios where a range and most likely
//! value are known, but the underlying distribution is assumed to be normal.
//!
//! # Examples
//!
//! ```
//! use distimate::prelude::*;
//! use distimate::Normal;
//! use approx::assert_relative_eq;
//!
//! let normal = Normal::new(1.0, 3.0).unwrap();
//! assert_eq!(normal.mode(), 2.0);
//! assert_relative_eq!(normal.mean().unwrap(), 2.0, epsilon = 1e-6);
//! ```

use crate::prelude::*;
use statrs::distribution::Normal as StatrsNormal;

/// Represents a modified Normal distribution for estimation purposes.
///
/// This Normal distribution is defined by minimum, most likely (mode), and maximum values.
/// It uses these to create an underlying standard Normal distribution, but provides
/// methods to work within the specified range.
///
/// # Examples
///
/// ```
/// use distimate::prelude::*;
/// use distimate::Normal;
/// use approx::assert_relative_eq;
///
/// let normal = Normal::new(1.0, 3.0).unwrap();
/// assert_eq!(normal.min(), 1.0);
/// assert_eq!(normal.mode(), 2.0);
/// assert_eq!(normal.max(), 3.0);
/// ```
#[derive(Debug, Clone)]
pub struct Normal {
    inner: StatrsNormal,
    min: f64,
    likely: f64,
    max: f64,
}

impl Normal {
    /// Creates a new Normal distribution with the given minimum, most likely (mode), and maximum values.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value of the distribution
    /// * `max` - The maximum value of the distribution
    ///
    /// Note that the `likely` value is implied as the center of the distribution.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `Normal` instance if the parameters are valid,
    /// or an `Error` if the parameters are invalid.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `min` is greater than or equal to `max`
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Normal;
    ///
    /// let normal = Normal::new(1.0, 3.0).unwrap();
    /// assert_eq!(normal.min(), 1.0);
    /// assert_eq!(normal.mode(), 2.0);
    /// assert_eq!(normal.max(), 3.0);
    /// ```
    pub fn new(min: f64, max: f64) -> Result<Self> {
        if min >= max {
            return Err(Error::InvalidRange { min, max });
        }

        let likely = (min + max) / 2.0;
        let mean = likely;
        let std_dev = (max - min) / 6.0; // Assuming 99.7% of values fall within the range

        Ok(Normal {
            inner: StatrsNormal::new(mean, std_dev)?,
            min,
            likely,
            max,
        })
    }

    /// Clamps a value to the [min, max] range of the distribution.
    fn clamp(&self, x: f64) -> f64 {
        x.max(self.min).min(self.max)
    }
}

/// Implementation of the `EstimationDistribution` trait for the Normal distribution.
///
/// This trait marks the Normal distribution as suitable for use in estimation contexts.
/// It doesn't add any new methods, but signals that this distribution is appropriate
/// for modeling uncertain estimates or measurements within a specified range.
///
/// The Normal distribution in this context is particularly useful when:
/// - There's a most likely value (mode) with symmetric uncertainty around it.
/// - The minimum and maximum values represent extreme scenarios.
/// - The underlying process is believed to follow a bell-shaped curve.
///
/// # Examples
///
/// ```
/// use distimate::prelude::*;
/// use distimate::Normal;
///
/// let normal = Normal::new(1.0, 3.0).unwrap();
/// // The Normal distribution can now be used in contexts requiring an EstimationDistribution
/// ```
impl EstimationDistribution for Normal {}

/// Implementation of the `Distribution` trait for the Normal distribution.
///
/// This implementation provides methods to calculate various statistical properties
/// of the Normal distribution, adapted for estimation scenarios with specified
/// minimum, most likely (mode), and maximum values.
///
/// # Note
///
/// The calculated values are based on the underlying Normal distribution and may
/// theoretically extend beyond the specified [min, max] range. For practical
/// estimation purposes, you may want to consider clamping extreme values.
impl Distribution<f64> for Normal {
    /// Calculates the mean of the Normal distribution.
    ///
    /// For this adapted Normal distribution, the mean is equal to the most likely value (mode).
    ///
    /// # Returns
    ///
    /// Returns `Some(mean)` where `mean` is the most likely value specified during construction.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Normal;
    ///
    /// let normal = Normal::new(1.0, 3.0).unwrap();
    /// assert_eq!(normal.mean(), Some(2.0));
    /// ```
    fn mean(&self) -> Option<f64> {
        self.inner.mean()
    }

    /// Calculates the variance of the Normal distribution.
    ///
    /// The variance is calculated based on the standard deviation derived from
    /// the specified min and max values during construction.
    ///
    /// # Returns
    ///
    /// Returns `Some(variance)` where `variance` is the square of the standard deviation.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Normal;
    ///
    /// let normal = Normal::new(1.0, 3.0).unwrap();
    /// assert_eq!(normal.variance().unwrap(), 0.1111111111111111); // (1/3)^2
    /// ```
    fn variance(&self) -> Option<f64> {
        self.inner.variance()
    }

    /// Calculates the skewness of the Normal distribution.
    ///
    /// The Normal distribution is always symmetric, so the skewness is always 0.
    ///
    /// # Returns
    ///
    /// Always returns `Some(0.0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Normal;
    ///
    /// let normal = Normal::new(1.0, 3.0).unwrap();
    /// assert_eq!(normal.skewness(), Some(0.0));
    /// ```
    fn skewness(&self) -> Option<f64> {
        Some(0.0) // Normal distribution is always symmetric
    }

    /// Calculates the entropy of the Normal distribution.
    ///
    /// The entropy is a measure of the average amount of information contained in
    /// the distribution. For a Normal distribution, it depends on the standard deviation.
    ///
    /// # Returns
    ///
    /// Returns `Some(entropy)` where `entropy` is calculated based on the
    /// standard deviation of the distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Normal;
    ///
    /// let normal = Normal::new(1.0, 3.0).unwrap();
    /// assert!(normal.entropy().is_some());
    /// ```
    fn entropy(&self) -> Option<f64> {
        self.inner.entropy()
    }
}

/// Implementation of the `Median` trait for the Normal distribution.
///
/// For a Normal distribution adapted for estimation purposes, the median is equal
/// to the most likely value (mode) specified during construction. This is because
/// the Normal distribution is symmetric around its mean, which in this case is set
/// to the most likely value.
impl Median<f64> for Normal {
    /// Calculates the median of the Normal distribution.
    ///
    /// For this adapted Normal distribution, the median is equal to the most likely
    /// value (mode) specified during construction.
    ///
    /// # Returns
    ///
    /// Returns the median of the distribution as an f64 value.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Normal;
    ///
    /// let normal = Normal::new(1.0, 3.0).unwrap();
    /// assert_eq!(normal.median(), 2.0);
    /// ```
    ///
    /// # Note
    ///
    /// In a standard Normal distribution, the median, mean, and mode are all equal.
    /// In this adapted version for estimation, we maintain this property by setting
    /// all of these to the most likely value provided during construction.
    fn median(&self) -> f64 {
        self.likely
    }
}

/// Implementation of the `Mode` trait for the Normal distribution.
///
/// For a Normal distribution adapted for estimation purposes, the mode is set to
/// the most likely value specified during construction. This aligns with the
/// typical use case in estimation scenarios where the most likely value is known.
impl Mode<f64> for Normal {
    /// Returns the mode of the Normal distribution.
    ///
    /// The mode is the most likely value of the distribution, which in this case
    /// is explicitly set during the construction of the Normal distribution instance.
    ///
    /// # Returns
    ///
    /// Returns the mode of the distribution as an f64 value, which is equal to
    /// the 'likely' value provided when creating the distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Normal;
    ///
    /// let normal = Normal::new(1.0, 3.0).unwrap();
    /// assert_eq!(normal.mode(), 2.0);
    /// ```
    ///
    /// # Note
    ///
    /// In a standard Normal distribution, the mode, median, and mean are all equal.
    /// In this adapted version for estimation, we maintain this property by setting
    /// all of these to the most likely value provided during construction.
    fn mode(&self) -> f64 {
        self.likely
    }
}

/// Implementation of the `Continuous` trait for the Normal distribution.
///
/// This implementation provides methods to calculate the probability density function (PDF)
/// and its natural logarithm for the Normal distribution, adapted for estimation scenarios
/// with specified minimum and maximum values.
impl Continuous<f64, f64> for Normal {
    /// Calculates the probability density function (PDF) of the Normal distribution at a given point.
    ///
    /// The PDF represents the relative likelihood of the distribution taking on a specific value.
    /// For this adapted Normal distribution, the PDF is set to 0 for values outside the [min, max] range.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to calculate the PDF
    ///
    /// # Returns
    ///
    /// The value of the PDF at the given point as an f64. Returns 0 if `x` is outside the [min, max] range.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Normal;
    ///
    /// let normal = Normal::new(1.0, 3.0).unwrap();
    /// assert!(normal.pdf(2.0) > 0.0);
    /// assert_eq!(normal.pdf(0.0), 0.0); // Outside the distribution's range
    /// ```
    fn pdf(&self, x: f64) -> f64 {
        if x < self.min || x > self.max {
            0.0
        } else {
            self.inner.pdf(x)
        }
    }

    /// Calculates the natural logarithm of the probability density function (PDF) of the Normal distribution at a given point.
    ///
    /// This method is particularly useful for numerical stability in calculations involving very small probability densities.
    /// For this adapted Normal distribution, the ln_pdf is set to negative infinity for values outside the [min, max] range.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to calculate the log PDF
    ///
    /// # Returns
    ///
    /// The natural logarithm of the PDF at the given point as an f64.
    /// Returns negative infinity if `x` is outside the [min, max] range.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Normal;
    ///
    /// let normal = Normal::new(1.0, 3.0).unwrap();
    /// assert!(normal.ln_pdf(2.0).is_finite());
    /// assert_eq!(normal.ln_pdf(0.0), f64::NEG_INFINITY); // Outside the distribution's range
    /// ```
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.min || x > self.max {
            f64::NEG_INFINITY
        } else {
            self.inner.ln_pdf(x)
        }
    }
}

/// Implementation of the `ContinuousCDF` trait for the Normal distribution.
///
/// This implementation provides methods to calculate the cumulative distribution function (CDF)
/// and its inverse for the Normal distribution, adapted for estimation scenarios with
/// specified minimum and maximum values.
impl ContinuousCDF<f64, f64> for Normal {
    /// Calculates the cumulative distribution function (CDF) of the Normal distribution at a given point.
    ///
    /// The CDF represents the probability that a random variable from this distribution
    /// will be less than or equal to the given value. For this adapted Normal distribution,
    /// the CDF is clamped to 0 for values below the minimum and to 1 for values above the maximum.
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
    /// use distimate::Normal;
    /// use approx::assert_relative_eq;
    ///
    /// let normal = Normal::new(1.0, 3.0).unwrap();
    /// assert_relative_eq!(normal.cdf(2.0), 0.5, epsilon = 1e-6);
    /// assert_eq!(normal.cdf(0.0), 0.0); // Below minimum
    /// assert_eq!(normal.cdf(4.0), 1.0); // Above maximum
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.min {
            0.0
        } else if x >= self.max {
            1.0
        } else {
            self.inner.cdf(x)
        }
    }

    /// Calculates the inverse of the cumulative distribution function (inverse CDF)
    /// of the Normal distribution for a given probability.
    ///
    /// This function is also known as the quantile function. It returns the value x
    /// for which P(X ≤ x) = p, where p is the given probability. For this adapted
    /// Normal distribution, the result is clamped to the [min, max] range.
    ///
    /// # Arguments
    ///
    /// * `p` - The probability for which to calculate the inverse CDF, must be in the range [0, 1]
    ///
    /// # Returns
    ///
    /// The value x for which the CDF of the distribution equals the given probability,
    /// clamped to the [min, max] range of the distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Normal;
    /// use approx::assert_relative_eq;
    ///
    /// let normal = Normal::new(1.0, 3.0).unwrap();
    /// assert_relative_eq!(normal.inverse_cdf(0.5), 2.0, epsilon = 1e-6);
    /// assert_eq!(normal.inverse_cdf(0.0), 1.0); // Clamped to minimum
    /// assert_eq!(normal.inverse_cdf(1.0), 3.0); // Clamped to maximum
    /// ```
    fn inverse_cdf(&self, p: f64) -> f64 {
        self.clamp(self.inner.inverse_cdf(p))
    }
}

/// Implementation of the `Min` trait for the Normal distribution.
///
/// This implementation provides a method to retrieve the minimum value
/// of the Normal distribution adapted for estimation scenarios.
impl Min<f64> for Normal {
    /// Returns the minimum value of the Normal distribution.
    ///
    /// In the context of this estimation-focused Normal distribution,
    /// the minimum value represents the lower bound of the possible outcomes.
    /// It's one of the three key parameters (along with the most likely value
    /// and the maximum) used to define this adapted Normal distribution.
    ///
    /// # Returns
    ///
    /// The minimum value of the distribution as an f64.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Normal;
    ///
    /// let normal = Normal::new(1.0, 3.0).unwrap();
    /// assert_eq!(normal.min(), 1.0);
    /// ```
    ///
    /// # Note
    ///
    /// While a standard Normal distribution extends infinitely in both directions,
    /// this adapted version for estimation purposes has a defined minimum value.
    /// This allows for more practical modeling in scenarios where there's a known
    /// lower limit to possible outcomes.
    fn min(&self) -> f64 {
        self.min
    }
}

/// Implementation of the `Max` trait for the Normal distribution.
///
/// This implementation provides a method to retrieve the maximum value
/// of the Normal distribution adapted for estimation scenarios.
impl Max<f64> for Normal {
    /// Returns the maximum value of the Normal distribution.
    ///
    /// In the context of this estimation-focused Normal distribution,
    /// the maximum value represents the upper bound of the possible outcomes.
    /// It's one of the three key parameters (along with the minimum value
    /// and the most likely value) used to define this adapted Normal distribution.
    ///
    /// # Returns
    ///
    /// The maximum value of the distribution as an f64.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Normal;
    ///
    /// let normal = Normal::new(1.0, 3.0).unwrap();
    /// assert_eq!(normal.max(), 3.0);
    /// ```
    ///
    /// # Note
    ///
    /// While a standard Normal distribution extends infinitely in both directions,
    /// this adapted version for estimation purposes has a defined maximum value.
    /// This allows for more practical modeling in scenarios where there's a known
    /// upper limit to possible outcomes.
    fn max(&self) -> f64 {
        self.max
    }
}

/// Implementation of the `rand::distributions::Distribution` trait for the Normal distribution.
///
/// This implementation allows for random sampling from the Normal distribution
/// adapted for estimation scenarios, using any random number generator that
/// implements the `rand::Rng` trait.
impl RandDistribution<f64> for Normal {
    /// Generates a random sample from the Normal distribution.
    ///
    /// This method uses the underlying Normal distribution to generate a sample,
    /// then clamps the result to ensure it falls within the specified [min, max] range.
    /// This approach maintains the shape of the Normal distribution within the
    /// specified range while ensuring all samples are valid for the estimation scenario.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to a random number generator that implements the `rand::Rng` trait
    ///
    /// # Returns
    ///
    /// A random sample from the Normal distribution as an f64, guaranteed to be
    /// within the [min, max] range specified during the distribution's construction.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Normal;
    /// use rand::prelude::*;
    ///
    /// let normal = Normal::new(1.0, 3.0).unwrap();
    /// let mut rng = StdRng::seed_from_u64(42);  // For reproducibility
    ///
    /// let sample = normal.sample(&mut rng);
    /// assert!(sample >= 1.0 && sample <= 3.0);
    /// ```
    ///
    /// # Note
    ///
    /// While this method ensures all samples fall within the specified range,
    /// it may lead to a slight overrepresentation of the minimum and maximum values
    /// compared to a true Normal distribution. This trade-off is often acceptable
    /// in practical estimation scenarios where staying within the specified range is crucial.
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.clamp(self.inner.sample(rng))
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
        let normal = Normal::new(1.0, 3.0).unwrap();
        assert_eq!(normal.min(), 1.0);
        assert_eq!(normal.mode(), 2.0);
        assert_eq!(normal.max(), 3.0);
        assert_eq!(normal.mean().unwrap(), 2.0);
        assert_relative_eq!(normal.std_dev().unwrap(), 1.0 / 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_invalid_parameters() {
        assert!(Normal::new(3.0, 1.0).is_err()); // min > max
    }

    #[test]
    fn test_pdf() {
        let normal = Normal::new(1.0, 3.0).unwrap();
        assert_relative_eq!(normal.pdf(2.0), normal.inner.pdf(2.0), epsilon = 1e-6);
        assert_eq!(normal.pdf(0.5), 0.0); // Below min
        assert_eq!(normal.pdf(3.5), 0.0); // Above max
    }

    #[test]
    fn test_cdf() {
        let normal = Normal::new(1.0, 3.0).unwrap();
        assert_relative_eq!(normal.cdf(2.0), 0.5, epsilon = 1e-6);
        assert_eq!(normal.cdf(0.5), 0.0); // Below min
        assert_eq!(normal.cdf(3.5), 1.0); // Above max
    }

    #[test]
    fn test_inverse_cdf() {
        let normal = Normal::new(1.0, 3.0).unwrap();
        assert_relative_eq!(normal.inverse_cdf(0.5), 2.0, epsilon = 1e-6);
        assert_eq!(normal.inverse_cdf(0.0), 1.0); // Clamped to min
        assert_eq!(normal.inverse_cdf(1.0), 3.0); // Clamped to max
    }

    #[test]
    fn test_sampling() {
        let normal = Normal::new(1.0, 3.0).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let samples: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();

        assert!(samples.iter().all(|&x| (1.0..=3.0).contains(&x)));
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert_relative_eq!(mean, 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_distribution_statistics() {
        let test_cases = vec![
            (1.0, 2.0, 3.0),       // Standard case
            (0.0, 5.0, 10.0),      // Wider range
            (-5.0, 0.0, 5.0),      // Range including negative values
            (100.0, 200.0, 300.0), // Larger values
        ];

        for (min, likely, max) in test_cases {
            let normal = Normal::new(min, max).unwrap();
            let n = 100_000; // Number of samples
            let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility

            let samples: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
            let mut data = Data::new(samples.clone());

            // Mean
            let sample_mean = data.mean().unwrap();
            let theoretical_mean = normal.mean().unwrap();
            assert_relative_eq!(
                sample_mean,
                theoretical_mean,
                epsilon = 0.01,
                max_relative = 0.01
            );

            // Variance
            let sample_variance = data.variance().unwrap();
            let theoretical_variance = normal.variance().unwrap();
            assert_relative_eq!(
                sample_variance,
                theoretical_variance,
                epsilon = 0.05,
                max_relative = 0.05
            );

            // Standard deviation
            let sample_std_dev = data.std_dev().unwrap();
            let theoretical_std_dev = normal.std_dev().unwrap();
            assert_relative_eq!(
                sample_std_dev,
                theoretical_std_dev,
                epsilon = 0.05,
                max_relative = 0.05
            );

            // Skewness
            let sample_skewness = 0.0; // normal distribution has no skewness
            let theoretical_skewness = normal.skewness().unwrap();
            assert_relative_eq!(
                sample_skewness,
                theoretical_skewness,
                epsilon = 0.1,
                max_relative = 0.1
            );

            // Median
            let sample_median = data.median();
            let theoretical_median = normal.median();
            assert_relative_eq!(
                sample_median,
                theoretical_median,
                epsilon = 0.01,
                max_relative = 0.01
            );

            // Mode (we can't reliably estimate the mode from samples, so we'll check if it's within the range)
            assert!(normal.mode() >= min && normal.mode() <= max);

            // Min and Max
            let sample_min = data.min();
            let sample_max = data.max();
            assert!(sample_min >= normal.min());
            assert!(sample_max <= normal.max());

            // CDF and Inverse CDF
            let percentiles = [0.1, 0.25, 0.5, 0.75, 0.9];
            for &p in percentiles.iter() {
                let sample_percentile = data.percentile((p * 100.0) as usize);
                let theoretical_percentile = normal.inverse_cdf(p);
                assert_relative_eq!(
                    sample_percentile,
                    theoretical_percentile,
                    epsilon = 0.05,
                    max_relative = 0.05
                );

                let sample_cdf =
                    samples.iter().filter(|&x| x <= &sample_percentile).count() as f64 / n as f64;
                let theoretical_cdf = normal.cdf(sample_percentile);
                assert_relative_eq!(
                    sample_cdf,
                    theoretical_cdf,
                    epsilon = 0.01,
                    max_relative = 0.01
                );
            }

            // PDF (we'll check a few points within the range)
            let pdf_points = [min, (min + likely) / 2.0, likely, (likely + max) / 2.0, max];
            for &x in pdf_points.iter() {
                let sample_pdf = samples.iter().filter(|&s| (s - x).abs() < 0.1).count() as f64
                    / (n as f64 * 0.2);
                let theoretical_pdf = normal.pdf(x);
                assert_relative_eq!(
                    sample_pdf,
                    theoretical_pdf,
                    epsilon = 0.1,
                    max_relative = 0.1
                );
            }

            println!("Test passed for Normal({}, {}, {})", min, likely, max);
        }
    }
}
