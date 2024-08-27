//! # Distimate: Estimation Distributions for Project Planning and Risk Analysis
//!
//! `distimate` is a Rust crate that provides a set of probability distributions
//! commonly used in estimation, along with a trait for working with these
//! distributions in an estimation context. It's designed to assist in project
//! planning, risk analysis, and decision-making under uncertainty.
//!
//! ## Features
//!
//! - A variety of estimation-focused probability distributions
//! - A unified `EstimationDistribution` trait for working with these distributions
//! - Tools for calculating confidence intervals, risk assessments, and fit quality
//! - Easy-to-use prelude module for convenient imports
//!
//! ## Quick Start
//!
//! To get started with `distimate`, add it to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! distimate = "0.1.0"  # replace with the current version
//! ```
//!
//! Then, in your Rust file:
//!
//! ```rust
//! use distimate::prelude::*;
//! use distimate::Pert;
//!
//! fn main() -> Result<()> {
//!     let pert = Pert::new(1.0, 2.0, 3.0)?;
//!     println!("Expected value: {}", pert.expected_value());
//!     println!("95% confidence interval: {:?}", pert.confidence_interval(0.95));
//!     Ok(())
//! }
//! ```
//!
//! ## Available Distributions
//!
//! - PERT (Program Evaluation and Review Technique)
//! - Triangular
//! - Normal
//!
//! ## The EstimationDistribution Trait
//!
//! The `EstimationDistribution` trait provides a unified interface for working
//! with probability distributions in an estimation context. It includes methods
//! for calculating percentiles, confidence intervals, and assessing risk.
//!
//! ## Error Handling
//!
//! This crate uses a custom `Error` type and `Result` alias for error handling.
//! Most methods that can fail will return a `Result<T, Error>`.
//!
//! ## Prelude
//!
//! For convenience, commonly used items are re-exported in the `prelude` module.
//! You can import all of these at once with `use distimate::prelude::*;`.

// Module declarations
mod distributions;
mod error;
mod estimation_distribution;
mod fit_quality;
pub mod prelude;
mod result;

// Re-exports
pub use error::Error;
pub use estimation_distribution::EstimationDistribution;
pub use fit_quality::EstimationFitQuality;
pub use result::Result;

pub use distributions::{Normal, Pert, Triangular};
