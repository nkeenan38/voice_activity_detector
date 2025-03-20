#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

mod error;
mod iterator;
mod label;
mod predict;
mod sample;
#[cfg(feature = "async")]
mod stream;
mod vad;

pub use error::Error;
pub use iterator::{IteratorExt, LabelIterator, PredictIterator, SegmentMergerIterator};
pub use label::LabeledAudio;
pub use sample::Sample;
#[cfg(feature = "async")]
pub use stream::{LabelStream, PredictStream, StreamExt};
pub use vad::{VoiceActivityDetector, VoiceActivityDetectorBuilder};
