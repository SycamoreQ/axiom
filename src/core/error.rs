use candle_core;
use thiserror::Error;

use crate::core::{device::Device, dtype::DType};

/*
Errors that might pop up during tensor ops
*/

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("shape mismatch in {op}: lhs {lhs:?} incompatible with rhs {rhs:?}")]
    ShapeMismatch {
        op: &'static str,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },

    #[error("rank mismatch in {op}: expected {expected}, got {got}")]
    RankMismatch {
        op: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("dtype mismatch in {op}: lhs {lhs:?} incompatible with rhs {rhs:?}")]
    DTypeMismatch {
        op: &'static str,
        lhs: DType,
        rhs: DType,
    }, // String until dtype.rs exists

    #[error("device mismatch in {op}: lhs {lhs} incompatible with rhs {rhs}")]
    DeviceMismatch {
        op: &'static str,
        lhs: String,
        rhs: String,
    },

    #[error("out of bounds in {op}: index {index} out of size {size}")]
    OutOfBounds {
        op: &'static str,
        index: usize,
        size: usize,
    },

    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("cuda error: {0}")]
    Cuda(String),

    #[error("internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, CoreError>;
