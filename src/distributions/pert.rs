//! PERT (Program Evaluation and Review Technique) distribution implementation.
//!
//! This module provides an implementation of the PERT distribution, which is commonly used
//! in project management and risk analysis. The PERT distribution is a special case of the
//! Beta distribution, defined by its minimum, most likely (mode), and maximum values.
//!
//! # Examples
//!
//! ```
//! use distimate::prelude::*;
//! use distimate::Pert;
//! use approx::assert_relative_eq;
//!
//! let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
//! assert_eq!(pert.mode(), 2.0);
//! assert_relative_eq!(pert.mean().unwrap(), 2.0, epsilon = 1e-6);
//! ```

use crate::prelude::*;
use statrs::distribution::Beta;

/// Represents a PERT (Program Evaluation and Review Technique) distribution.
///
/// The PERT distribution is defined by its minimum, most likely (mode), and maximum values,
/// along with an optional shape parameter.
///
/// # Examples
///
/// ```
/// use distimate::prelude::*;
/// use distimate::Pert;
/// use approx::assert_relative_eq;
///
/// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
/// assert_eq!(pert.min(), 1.0);
/// assert_eq!(pert.mode(), 2.0);
/// assert_eq!(pert.max(), 3.0);
/// assert_relative_eq!(pert.mean().unwrap(), 2.0, epsilon = 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct Pert {
    min: f64,
    mode: f64,
    max: f64,
    shape: f64,
    inner: Beta,
}

#[allow(dead_code)]
impl Pert {
    /// Creates a new PERT distribution with the given minimum, likely, and maximum values.
    ///
    /// This method uses the default shape parameter of 4.0.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value of the distribution
    /// * `likely` - The most likely value (mode) of the distribution
    /// * `max` - The maximum value of the distribution
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `Pert` instance if the parameters are valid,
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
    /// use distimate::Pert;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_eq!(pert.min(), 1.0);
    /// assert_eq!(pert.mode(), 2.0);
    /// assert_eq!(pert.max(), 3.0);
    /// ```
    pub fn new(min: f64, likely: f64, max: f64) -> Result<Self> {
        Self::new_with_shape(min, likely, max, 4.0)
    }

    /// Creates a new PERT distribution with the given minimum, likely, maximum, and shape values.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value of the distribution
    /// * `likely` - The most likely value (mode) of the distribution
    /// * `max` - The maximum value of the distribution
    /// * `shape` - The shape parameter of the distribution (must be between 2.0 and 6.0, inclusive)
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `Pert` instance if the parameters are valid,
    /// or an `Error` if the parameters are invalid.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `min` is greater than or equal to `max`
    /// - `likely` is less than `min` or greater than `max`
    /// - `shape` is not between 2.0 and 6.0 (inclusive)
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    ///
    /// let pert = Pert::new_with_shape(1.0, 2.0, 3.0, 5.0).unwrap();
    /// assert_eq!(pert.min(), 1.0);
    /// assert_eq!(pert.mode(), 2.0);
    /// assert_eq!(pert.max(), 3.0);
    /// ```
    pub fn new_with_shape(min: f64, likely: f64, max: f64, shape: f64) -> Result<Self> {
        // validate inputs
        if min >= max {
            return Err(Error::InvalidRange { min, max });
        }
        if likely < min || likely > max {
            return Err(Error::InvalidMode { min, likely, max });
        }
        if !(2.0..=6.0).contains(&shape) {
            return Err(Error::InvalidShape(shape));
        }

        let alpha = 1.0 + shape * ((likely - min) / (max - min));
        let beta = 1.0 + shape * ((max - likely) / (max - min));

        Ok(Pert {
            min,
            mode: likely,
            max,
            shape,
            inner: Beta::new(alpha, beta)?,
        })
    }

    /// Returns the alpha parameter of the underlying Beta distribution.
    ///
    /// The alpha parameter is calculated based on the minimum, mode, maximum,
    /// and shape parameters of the PERT distribution.
    ///
    /// # Returns
    ///
    /// The alpha parameter as an f64 value.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.alpha(), 3.0, epsilon = 1e-6);
    /// ```
    pub fn alpha(&self) -> f64 {
        self.inner.shape_a()
    }

    /// Returns the beta parameter of the underlying Beta distribution.
    ///
    /// The beta parameter is calculated based on the minimum, mode, maximum,
    /// and shape parameters of the PERT distribution.
    ///
    /// # Returns
    ///
    /// The beta parameter as an f64 value.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.beta(), 3.0, epsilon = 1e-6);
    /// ```
    pub fn beta(&self) -> f64 {
        self.inner.shape_b()
    }

    /// Returns the shape parameter of the PERT distribution.
    ///
    /// The shape parameter influences the peakedness of the distribution.
    /// A higher value results in a more peaked distribution around the mode.
    ///
    /// # Returns
    ///
    /// The shape parameter as an f64 value.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    ///
    /// let pert = Pert::new_with_shape(1.0, 2.0, 3.0, 5.0).unwrap();
    /// assert_eq!(pert.shape(), 5.0);
    /// ```
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Calculates the excess kurtosis of the PERT distribution.
    ///
    /// Excess kurtosis is a measure of the "tailedness" of the probability distribution
    /// compared to a normal distribution. A positive excess kurtosis indicates heavier tails
    /// and a higher, sharper peak, while a negative excess kurtosis indicates lighter tails
    /// and a lower, more rounded peak.
    ///
    /// # Returns
    ///
    /// The excess kurtosis as an f64 value.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.kurtosis(), -0.6666667, epsilon = 1e-6);
    /// ```
    pub fn kurtosis(&self) -> f64 {
        let (a, b) = (self.alpha(), self.beta());
        (6.0 * ((a - b).powi(2) * (a + b + 1.0) - a * b * (a + b + 2.0)))
            / (a * b * (a + b + 2.0) * (a + b + 3.0))
    }
}

/// Implementation of the `EstimationDistribution` trait for the PERT distribution.
///
/// This trait marks the PERT distribution as suitable for use in estimation contexts.
/// It doesn't add any new methods, but signals that this distribution is appropriate
/// for modeling uncertain estimates or durations.
///
/// # Examples
///
/// ```
/// use distimate::prelude::*;
/// use distimate::Pert;
///
/// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
/// // The fact that we can create a Pert instance demonstrates
/// // that it implements EstimationDistribution
/// ```
impl EstimationDistribution for Pert {}

/// Implementation of the `Distribution` trait for the PERT distribution.
///
/// This implementation provides methods to calculate various statistical properties
/// of the PERT distribution.
impl Distribution<f64> for Pert {
    /// Calculates the mean of the PERT distribution.
    ///
    /// # Returns
    ///
    /// The mean of the distribution as an `Option<f64>`. Always returns `Some(value)`
    /// as the PERT distribution always has a defined mean.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.mean().unwrap(), 2.0, epsilon = 1e-6);
    /// ```
    fn mean(&self) -> Option<f64> {
        self.inner
            .mean()
            .map(|m| self.min + (self.max - self.min) * m)
    }

    /// Calculates the variance of the PERT distribution.
    ///
    /// # Returns
    ///
    /// The variance of the distribution as an `Option<f64>`. Always returns `Some(value)`
    /// as the PERT distribution always has a defined variance.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.variance().unwrap(), 1.0 / 7.0, epsilon = 1e-6);
    /// ```
    fn variance(&self) -> Option<f64> {
        self.inner
            .variance()
            .map(|v| (self.max - self.min).powi(2) * v)
    }

    /// Calculates the skewness of the PERT distribution.
    ///
    /// # Returns
    ///
    /// The skewness of the distribution as an `Option<f64>`. Always returns `Some(value)`
    /// as the PERT distribution always has a defined skewness.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.skewness().unwrap(), 0.0, epsilon = 1e-6);
    /// ```
    fn skewness(&self) -> Option<f64> {
        self.inner.skewness()
    }

    /// Calculates the entropy of the Pert distribution.
    ///
    /// The entropy is a measure of the average amount of information contained in the distribution.
    ///
    /// # Returns
    ///
    /// The entropy of the distribution as an `Option<f64>`. Always returns `Some(value)`
    /// as the Pert distribution always has a defined entropy.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.entropy().unwrap(), -0.267864, epsilon = 1e-6);
    /// ```
    fn entropy(&self) -> Option<f64> {
        self.inner.entropy()
    }
}

/// Implementation of the `Median` trait for the PERT distribution.
///
/// The median of a PERT distribution is the value that separates the lower and upper halves
/// of the probability distribution. For a PERT distribution, this is calculated using
/// the inverse cumulative distribution function (inverse CDF) at probability 0.5.
impl Median<f64> for Pert {
    /// Calculates the median of the PERT distribution.
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
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.median(), 2.0, epsilon = 1e-4);
    /// ```
    fn median(&self) -> f64 {
        self.min + (self.max - self.min) * self.inner.inverse_cdf(0.5)
    }
}

/// Implementation of the `Mode` trait for the PERT distribution.
impl Mode<f64> for Pert {
    /// Returns the mode of the PERT distribution.
    ///
    /// The mode is the most likely value of the distribution, which is one of the
    /// defining parameters of the PERT distribution.
    ///
    /// # Returns
    ///
    /// The mode of the distribution as an f64 value.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_eq!(pert.mode(), 2.0);
    /// ```
    fn mode(&self) -> f64 {
        self.mode
    }
}

/// Implementation of the `Continuous` trait for the PERT distribution.
///
/// This implementation provides methods to calculate the probability density function (PDF)
/// and its natural logarithm for the PERT distribution. These functions are crucial for
/// understanding the likelihood of different outcomes within the distribution's range.
///
/// The PERT distribution is continuous over its defined range (minimum to maximum),
/// with the PDF representing the relative likelihood of each possible value.
impl Continuous<f64, f64> for Pert {
    /// Calculates the probability density function (PDF) of the PERT distribution at a given point.
    ///
    /// The PDF represents the relative likelihood of the distribution taking on a specific value.
    /// For the PERT distribution, the PDF is non-zero only within the range [min, max] and
    /// reaches its peak at the mode.
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
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.pdf(2.0), 0.9375, epsilon = 1e-6);
    /// assert_eq!(pert.pdf(0.0), 0.0); // Outside the distribution's range
    /// ```
    fn pdf(&self, x: f64) -> f64 {
        if x < self.min || x > self.max {
            0.0
        } else {
            let scaled_x = (x - self.min) / (self.max - self.min);
            self.inner.pdf(scaled_x) / (self.max - self.min)
        }
    }

    /// Calculates the natural logarithm of the probability density function (PDF) of the PERT distribution at a given point.
    ///
    /// This method is particularly useful for numerical stability in calculations involving very small probability densities.
    /// It's often used in statistical inference and machine learning algorithms.
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
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.ln_pdf(2.0), -0.064538, epsilon = 1e-6);
    /// assert_eq!(pert.ln_pdf(0.0), f64::NEG_INFINITY); // Outside the distribution's range
    /// ```
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.min || x > self.max {
            f64::NEG_INFINITY
        } else {
            let scaled_x = (x - self.min) / (self.max - self.min);
            self.inner.ln_pdf(scaled_x) - (self.max - self.min).ln()
        }
    }
}

/// Implementation of the `ContinuousCDF` trait for the PERT distribution.
///
/// This implementation provides methods to calculate the cumulative distribution function (CDF)
/// and its inverse for the PERT distribution. These functions are essential for understanding
/// the probability of the distribution taking on a value less than or equal to a given point,
/// and for generating random variates from the distribution.
impl ContinuousCDF<f64, f64> for Pert {
    /// Calculates the cumulative distribution function (CDF) of the PERT distribution at a given point.
    ///
    /// The CDF represents the probability that a random variable from this distribution
    /// will be less than or equal to the given value.
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
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.cdf(2.0), 0.5, epsilon = 1e-6);
    /// assert_eq!(pert.cdf(1.0), 0.0);
    /// assert_eq!(pert.cdf(3.0), 1.0);
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.min {
            0.0
        } else if x >= self.max {
            1.0
        } else {
            let scaled_x = (x - self.min) / (self.max - self.min);
            self.inner.cdf(scaled_x)
        }
    }

    /// Calculates the inverse of the cumulative distribution function (inverse CDF)
    /// of the PERT distribution for a given probability.
    ///
    /// This function is also known as the quantile function. It returns the value x
    /// for which P(X ≤ x) = p, where p is the given probability.
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
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.inverse_cdf(0.5), 2.0, epsilon = 1e-4);
    /// assert_relative_eq!(pert.inverse_cdf(0.0), 1.0, epsilon = 1e-4);
    /// assert_relative_eq!(pert.inverse_cdf(1.0), 3.0, epsilon = 1e-4);
    /// ```
    fn inverse_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            self.min
        } else if p >= 1.0 {
            self.max
        } else {
            self.min + (self.max - self.min) * self.inner.inverse_cdf(p)
        }
    }
}

/// Implementation of the `Min` trait for the PERT distribution.
///
/// This implementation provides a method to retrieve the minimum value
/// of the PERT distribution, which is one of its defining parameters.
impl Min<f64> for Pert {
    /// Returns the minimum value of the PERT distribution.
    ///
    /// The minimum value represents the lowest possible outcome in the distribution.
    ///
    /// # Returns
    ///
    /// The minimum value of the distribution as an f64.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_eq!(pert.min(), 1.0);
    /// ```
    fn min(&self) -> f64 {
        self.min
    }
}

/// Implementation of the `Max` trait for the PERT distribution.
///
/// This implementation provides a method to retrieve the maximum value
/// of the PERT distribution, which is one of its defining parameters.
impl Max<f64> for Pert {
    /// Returns the maximum value of the PERT distribution.
    ///
    /// The maximum value represents the highest possible outcome in the distribution.
    ///
    /// # Returns
    ///
    /// The maximum value of the distribution as an f64.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_eq!(pert.max(), 3.0);
    /// ```
    fn max(&self) -> f64 {
        self.max
    }
}

/// Implementation of the `rand::distributions::Distribution` trait for the PERT distribution.
///
/// This implementation allows for random sampling from the PERT distribution using
/// any random number generator that implements the `rand::Rng` trait.
impl RandDistribution<f64> for Pert {
    /// Generates a random sample from the PERT distribution.
    ///
    /// This method uses the inverse transform sampling technique: it generates a uniform
    /// random number between 0 and 1, then applies the inverse CDF to this number to
    /// produce a sample from the PERT distribution.
    ///
    /// # Arguments
    ///
    /// * `rng` - A random number generator that implements the `rand::Rng` trait
    ///
    /// # Returns
    ///
    /// A random sample from the PERT distribution as an f64.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    /// use rand::prelude::*;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// let mut rng = StdRng::seed_from_u64(42);  // For reproducibility
    ///
    /// let sample = pert.sample(&mut rng);
    /// assert!(sample >= 1.0 && sample <= 3.0);
    /// ```
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u: f64 = rng.gen();
        self.inverse_cdf(u)
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
        let pert = Pert::new_with_shape(1.0, 2.0, 3.0, 4.0).unwrap();
        assert_eq!(pert.min(), 1.0);
        assert_eq!(pert.mode(), 2.0);
        assert_eq!(pert.max(), 3.0);
    }

    #[test]
    fn test_min() {
        let pert = Pert::new_with_shape(1.0, 2.0, 3.0, 4.0).unwrap();
        assert_eq!(pert.min(), 1.0);
    }

    #[test]
    fn test_max() {
        let pert = Pert::new_with_shape(1.0, 2.0, 3.0, 4.0).unwrap();
        assert_eq!(pert.max(), 3.0);
    }

    #[test]
    fn test_alpha_beta() {
        let pert = Pert::new_with_shape(1.0, 2.0, 3.0, 4.0).unwrap();
        assert_relative_eq!(pert.alpha(), 3.0, epsilon = 1e-6);
        assert_relative_eq!(pert.beta(), 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mean() {
        let pert = Pert::new_with_shape(1.0, 2.0, 3.0, 4.0).unwrap();
        assert_relative_eq!(pert.mean().unwrap(), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_median() {
        let pert = Pert::new_with_shape(1.0, 2.0, 3.0, 4.0).unwrap();
        assert_relative_eq!(pert.median(), 2.0, epsilon = 1e-3);
    }

    #[test]
    fn test_mode() {
        let pert = Pert::new_with_shape(1.0, 2.0, 3.0, 4.0).unwrap();
        assert_eq!(pert.mode(), 2.0);
    }

    #[test]
    fn test_variance() {
        let pert = Pert::new_with_shape(1.0, 2.0, 3.0, 4.0).unwrap();
        assert_relative_eq!(pert.variance().unwrap(), 1.0 / 7.0, epsilon = 1e-6);
    }

    #[test]
    fn test_skewness() {
        let pert = Pert::new_with_shape(1.0, 2.0, 3.0, 4.0).unwrap();
        assert_relative_eq!(pert.skewness().unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_kurtosis() {
        let pert = Pert::new_with_shape(1.0, 2.0, 3.0, 4.0).unwrap();
        assert_relative_eq!(pert.kurtosis(), -0.66666667, epsilon = 1e-6);
    }

    #[test]
    fn test_asymmetric_distribution() {
        let pert = Pert::new_with_shape(0.0, 1.0, 5.0, 4.0).unwrap();
        assert_relative_eq!(pert.alpha(), 1.8, epsilon = 1e-6);
        assert_relative_eq!(pert.beta(), 4.2, epsilon = 1e-6);
        assert_relative_eq!(pert.mean().unwrap(), 1.5, epsilon = 1e-6);
        assert_relative_eq!(pert.median(), 1.375, epsilon = 1e-2);
        assert_relative_eq!(pert.variance().unwrap(), 0.75, epsilon = 1e-6);
        // assert_relative_eq!(pert.skewness().unwrap(), 0.5773502, epsilon = 1e-6);
        assert_relative_eq!(pert.kurtosis(), -0.2222222, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: InvalidRange { min: 3.0, max: 1.0 }"
    )]
    fn test_invalid_parameters() {
        Pert::new_with_shape(3.0, 2.0, 1.0, 4.0).unwrap(); // min > max
    }

    fn sample_statistics(dist: &Pert, n: usize) -> (f64, f64, f64, f64) {
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
    fn test_pert_distribution_statistics() {
        let min = 1.0;
        let mode = 2.0;
        let max = 5.0;
        let shape = 4.0;
        let dist = Pert::new_with_shape(min, mode, max, shape).unwrap();
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
    fn test_pert_distribution_bounds() {
        let min = 1.0;
        let mode = 2.0;
        let max = 3.0;
        let shape = 4.0;
        let dist = Pert::new_with_shape(min, mode, max, shape).unwrap();
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

    fn sample_kurtosis(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let m2 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let m4 = data.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
        m4 / m2.powi(2) - 3.0
    }

    #[test]
    fn test_distribution_statistics() {
        let test_cases = vec![
            (0.0, 5.0, 10.0, 4.0),       // Standard PERT
            (0.0, 2.0, 10.0, 2.0),       // Modified PERT with lower shape
            (0.0, 9.0, 10.0, 6.0),       // Modified PERT with higher shape
            (-10.0, 0.0, 10.0, 4.0),     // Standard PERT with negative values
            (100.0, 400.0, 1000.0, 4.0), // Standard PERT with larger values
        ];

        for (min, mode, max, shape) in test_cases {
            let pert = Pert::new_with_shape(min, mode, max, shape).unwrap();
            let n = 25_000; // Number of samples
            let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility

            let samples: Vec<f64> = (0..n).map(|_| pert.sample(&mut rng)).collect();
            let mut data = Data::new(samples.clone());

            // Mean
            let sample_mean = data.mean().unwrap();
            let theoretical_mean = pert.mean().unwrap();
            assert_relative_eq!(
                sample_mean,
                theoretical_mean,
                epsilon = 0.1,
                max_relative = 0.1
            );

            // Variance
            let sample_variance = data.variance().unwrap();
            let theoretical_variance = pert.variance().unwrap();
            assert_relative_eq!(
                sample_variance,
                theoretical_variance,
                epsilon = 0.05,
                max_relative = 0.05
            );

            // Standard deviation
            let sample_std_dev = data.std_dev().unwrap();
            let theoretical_std_dev = pert.std_dev().unwrap();
            assert_relative_eq!(
                sample_std_dev,
                theoretical_std_dev,
                epsilon = 0.1,
                max_relative = 0.1
            );

            // Skewness
            let sample_skewness = sample_skewness(&samples);
            let theoretical_skewness = pert.skewness().unwrap();
            assert_relative_eq!(
                sample_skewness,
                theoretical_skewness,
                epsilon = 0.1,
                max_relative = 0.1
            );

            // Median
            let sample_median = data.median();
            let theoretical_median = pert.median();
            assert_relative_eq!(
                sample_median,
                theoretical_median,
                epsilon = 0.1,
                max_relative = 0.1
            );

            // Mode (we can't reliably estimate the mode from samples, so we'll check if it's within the range)
            assert!(pert.mode() >= min && pert.mode() <= max);

            // Min and Max
            let sample_min = data.min();
            let sample_max = data.max();
            assert!(sample_min >= pert.min());
            assert!(sample_max <= pert.max());

            // Excess Kurtosis
            let sample_kurtosis = sample_kurtosis(&samples);
            let theoretical_kurtosis = pert.kurtosis();
            assert_relative_eq!(
                sample_kurtosis,
                theoretical_kurtosis,
                epsilon = 0.1,
                max_relative = 0.1
            );

            // CDF and Inverse CDF
            let percentiles = [0.1, 0.25, 0.5, 0.75, 0.9];
            for p in percentiles.iter() {
                let sample_percentile = data.percentile((100.0 * p) as usize);
                let theoretical_percentile = pert.inverse_cdf(*p);
                assert_relative_eq!(
                    sample_percentile,
                    theoretical_percentile,
                    epsilon = 0.05,
                    max_relative = 0.05
                );

                let sample_cdf =
                    samples.iter().filter(|&x| x <= &sample_percentile).count() as f64 / n as f64;
                let theoretical_cdf = pert.cdf(sample_percentile);
                assert_relative_eq!(
                    sample_cdf,
                    theoretical_cdf,
                    epsilon = 0.01,
                    max_relative = 0.01
                );
            }

            println!(
                "Test passed for PERT({}, {}, {}, {})",
                min, mode, max, shape
            );
        }
    }
}
