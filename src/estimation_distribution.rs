use rand::distributions::Distribution as RandDistribution;
use statrs::{
    distribution::{Continuous, ContinuousCDF},
    statistics::{Distribution, Max, Median, Min, Mode},
};

use crate::{Error, EstimationFitQuality, Result};

/// A trait for probability distributions used in estimation scenarios.
///
/// This trait combines several statistical traits and provides additional methods
/// specific to estimation and risk analysis. It's designed to be implemented by
/// distributions that model uncertainty in estimates, such as project durations,
/// costs, or other quantifiable outcomes.
///
/// # Examples
///
/// ```
/// use distimate::prelude::*;
/// use distimate::Pert;
/// use approx::assert_relative_eq;
///
/// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
/// assert_relative_eq!(pert.expected_value(), 2.0, epsilon = 1e-6);
/// assert_relative_eq!(pert.optimistic_estimate(), 1.37847, epsilon = 1e-4);
/// assert_relative_eq!(pert.pessimistic_estimate(), 2.62152, epsilon = 1e-4);
/// ```
pub trait EstimationDistribution:
    Distribution<f64>
    + Continuous<f64, f64>
    + ContinuousCDF<f64, f64>
    + Median<f64>
    + Mode<f64>
    + Min<f64>
    + Max<f64>
    + RandDistribution<f64> // TODO: add TryFrom (x2) here
{
    /// Returns the estimate at a given percentile.
    ///
    /// # Arguments
    ///
    /// * `p` - The percentile as a float between 0.0 and 1.0.
    ///
    /// # Returns
    ///
    /// The estimated value at the given percentile, or an error if the percentile is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.percentile_estimate(0.5).unwrap(), 2.0, epsilon = 1e-4);
    /// ```
    fn percentile_estimate(&self, p: f64) -> Result<f64> {
        if !(0.0..=1.0).contains(&p) {
            Err(Error::InvalidPercentile(p))
        } else {
            Ok(self.inverse_cdf(p))
        }
    }

    /// Returns the optimistic (best-case) estimate, typically the 5th percentile.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.optimistic_estimate(), 1.37847, epsilon = 1e-4);
    /// ```
    fn optimistic_estimate(&self) -> f64 {
        self.percentile_estimate(0.05).unwrap()
    }

    /// Returns the pessimistic (worst-case) estimate, typically the 95th percentile.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.pessimistic_estimate(), 2.62152, epsilon = 1e-4);
    /// ```
    fn pessimistic_estimate(&self) -> f64 {
        self.percentile_estimate(0.95).unwrap()
    }

    /// Returns the most likely estimate (mode).
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_eq!(pert.most_likely_estimate(), 2.0);
    /// ```
    fn most_likely_estimate(&self) -> f64 {
        self.mode()
    }

    /// Returns the expected value (mean) of the distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.expected_value(), 2.0, epsilon = 1e-6);
    /// ```
    fn expected_value(&self) -> f64 {
        self.mean().unwrap()
    }

    /// Returns the standard deviation of the distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use distimate::prelude::*;
    /// use distimate::Pert;
    /// use approx::assert_relative_eq;
    ///
    /// let pert = Pert::new(1.0, 2.0, 3.0).unwrap();
    /// assert_relative_eq!(pert.uncertainty(), 0.377964, epsilon = 1e-4);
    /// ```
    fn uncertainty(&self) -> f64 {
        self.std_dev().unwrap()
    }

    fn probability_of_completion(&self, estimate: f64) -> f64 {
        self.cdf(estimate)
    }

    /// Calculates the risk of exceeding a given estimate.
    ///
    /// This method computes the probability that the actual value will be greater than
    /// the provided estimate. It's particularly useful in project management and risk
    /// analysis to quantify the likelihood of cost overruns or schedule delays.
    ///
    /// The risk of overrun is calculated as 1 minus the probability of completion:
    /// Risk = 1 - P(X <= estimate), where X is the random variable representing the
    /// actual cost or time.
    ///
    /// For example:
    /// - A risk of 0.2 (20%) means there's a 20% chance the project will exceed the estimate.
    /// - A risk of 0.5 (50%) means it's equally likely to be above or below the estimate.
    /// - A risk of 0.8 (80%) suggests a high likelihood of exceeding the estimate.
    ///
    /// # Arguments
    ///
    /// * `estimate` - The point estimate to evaluate the risk against.
    ///
    /// # Returns
    ///
    /// A float between 0 and 1 representing the probability of exceeding the estimate.
    fn risk_of_overrun(&self, estimate: f64) -> f64 {
        1.0 - self.probability_of_completion(estimate)
    }

    /// Returns a confidence interval for the estimate.
    ///
    /// A confidence interval provides a range of values that is likely to contain
    /// the true value with a certain level of confidence. It's useful for expressing
    /// the uncertainty in an estimate.
    ///
    /// This method calculates a two-sided confidence interval, meaning it provides
    /// both a lower and upper bound. The interval is centered around the median of
    /// the distribution.
    ///
    /// # Arguments
    ///
    /// * `confidence_level` - A value between 0 and 1 representing the desired level of confidence.
    ///   Common values are 0.95 for a 95% confidence interval or 0.99 for a 99% confidence interval.
    ///
    /// # Returns
    ///
    /// A tuple (lower, upper) representing the lower and upper bounds of the confidence interval.
    ///
    /// # Example
    ///
    /// If you call `confidence_interval(0.95)` and get (100, 150), it means:
    /// "We are 95% confident that the true value lies between 100 and 150."
    ///
    /// # Note
    ///
    /// The width of the interval indicates the precision of the estimate. A narrower interval
    /// suggests a more precise estimate, while a wider interval indicates more uncertainty.
    fn confidence_interval(&self, confidence_level: f64) -> (f64, f64) {
        let lower = self.percentile_estimate((1.0 - confidence_level) / 2.0);
        let upper = self.percentile_estimate(1.0 - (1.0 - confidence_level) / 2.0);
        (lower.unwrap(), upper.unwrap())
    }

    /// Evaluates how well the distribution fits a given dataset in the context of estimation.
    ///
    /// This method provides a measure of how accurately the distribution represents
    /// the provided set of estimates or actual values.
    ///
    /// # Arguments
    ///
    /// * `data` - A slice of f64 values representing historical estimates or actual values.
    ///
    /// # Returns
    ///
    /// Returns an `EstimationFitQuality` struct containing various fit metrics.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if there's an issue calculating the fit,
    /// such as invalid data points or errors in percentile calculations.
    fn evaluate_fit_quality(&self, data: &[f64]) -> Result<EstimationFitQuality> {
        let n = data.len();
        if n == 0 {
            return Err(Error::InsufficientData {
                required: 1,
                provided: 0,
            });
        }

        let mean = data.iter().sum::<f64>() / n as f64;
        let median = self.median();
        let mode = self.mode();

        let mse = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let rmse = mse.sqrt();

        let mape = data.iter().map(|&x| ((x - mean) / x).abs()).sum::<f64>() / n as f64 * 100.0;

        let within_50 = self.calculate_within_interval(data, 0.25, 0.75)?;
        let within_68 = self.calculate_within_interval(data, 0.16, 0.84)?;
        let within_95 = self.calculate_within_interval(data, 0.025, 0.975)?;

        Ok(EstimationFitQuality {
            mean_absolute_percentage_error: mape,
            root_mean_square_error: rmse,
            distribution_mean: mean,
            distribution_median: median,
            distribution_mode: mode,
            within_50_percent: within_50,
            within_68_percent: within_68,
            within_95_percent: within_95,
            sample_size: n,
        })
    }

    /// Helper method to calculate the proportion of data within a given interval.
    ///
    /// # Arguments
    ///
    /// * `data` - A slice of f64 values representing the dataset.
    /// * `lower_percentile` - The lower bound of the interval as a percentile (0.0 to 1.0).
    /// * `upper_percentile` - The upper bound of the interval as a percentile (0.0 to 1.0).
    ///
    /// # Returns
    ///
    /// Returns the proportion of data points within the specified interval.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if there's an issue calculating the percentiles.
    fn calculate_within_interval(
        &self,
        data: &[f64],
        lower_percentile: f64,
        upper_percentile: f64,
    ) -> Result<f64> {
        let lower_bound = self
            .percentile_estimate(lower_percentile)
            .map_err(|e| Error::FitQualityError(format!("Error calculating lower bound: {}", e)))?;
        let upper_bound = self
            .percentile_estimate(upper_percentile)
            .map_err(|e| Error::FitQualityError(format!("Error calculating upper bound: {}", e)))?;

        let count_within = data
            .iter()
            .filter(|&&x| x >= lower_bound && x <= upper_bound)
            .count();

        Ok(count_within as f64 / data.len() as f64)
    }
}
