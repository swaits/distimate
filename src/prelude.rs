pub use crate::{Error, EstimationDistribution, EstimationFitQuality, Result};

pub use rand::distributions::Distribution as RandDistribution;

pub use statrs::{
    distribution::{Beta, Continuous, ContinuousCDF},
    statistics::{Distribution, Max, Median, Min, Mode},
};
