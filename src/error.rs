use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Insufficient data: at least {required} data points required, but got {provided}")]
    InsufficientData { required: usize, provided: usize },

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Fitting error: {0}")]
    FittingError(String),

    #[error("Fit quality check error: {0}")]
    FitQualityError(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Math error: {0}")]
    MathError(String),

    #[error("Invalid percentile: {0}. Must be between 0 and 1.")]
    InvalidPercentile(f64),

    #[error("Invalid confidence level: {0}. Must be between 0 and 1.")]
    InvalidConfidenceLevel(f64),

    #[error("Invalid shape: {0}. Must be between 0 and 1.")]
    InvalidShape(f64),

    #[error("Invalid parameters: min ({min}) must be less than max ({max}).")]
    InvalidRange { min: f64, max: f64 },

    #[error("Likely value ({likely}) must be between min ({min}) and max ({max}).")]
    InvalidMode { min: f64, likely: f64, max: f64 },

    #[error("statrs::StatsError: {0}.")]
    StatsError(#[from] statrs::StatsError),
}
