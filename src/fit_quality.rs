/// Represents the quality of fit for an estimation distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct EstimationFitQuality {
    /// Mean Absolute Percentage Error (MAPE)
    pub mean_absolute_percentage_error: f64,
    /// Root Mean Square Error (RMSE)
    pub root_mean_square_error: f64,
    /// Mean of the fitted distribution
    pub distribution_mean: f64,
    /// Median of the fitted distribution
    pub distribution_median: f64,
    /// Mode of the fitted distribution
    pub distribution_mode: f64,
    /// Proportion of data within 50% confidence interval (interquartile range)
    pub within_50_percent: f64,
    /// Proportion of data within 68% confidence interval (approx. 1 std dev)
    pub within_68_percent: f64,
    /// Proportion of data within 95% confidence interval (approx. 2 std dev)
    pub within_95_percent: f64,
    /// The number of data points used in the calculation
    pub sample_size: usize,
}
