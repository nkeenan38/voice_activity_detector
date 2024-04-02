#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

mod dynamic_vad;
mod error;
mod iterator;
mod label;
mod predict;
mod sample;
#[cfg(feature = "async")]
mod stream;
mod vad;

pub use dynamic_vad::DynamicVoiceActivityDetector;
pub use error::Error;
pub use iterator::{IteratorExt, LabelIterator, PredictIterator};
pub use label::LabeledAudio;
pub use sample::Sample;
#[cfg(feature = "async")]
pub use stream::{LabelStream, PredictStream, StreamExt};
pub use vad::VoiceActivityDetector;
