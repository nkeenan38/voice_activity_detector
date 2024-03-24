use crate::label::LabelState;
use crate::predict::PredictState;
use crate::{LabelIterator, PredictIterator, Sample, VoiceActivityDetector};

/// Extensions for iterators.
pub trait IteratorExt: Iterator {
    /// Creates a new [PredictIterator] from an iterator of samples.
    fn predict<const N: usize>(
        self,
        vad: VoiceActivityDetector<N>,
    ) -> PredictIterator<Self::Item, Self, N>
    where
        Self::Item: Sample,
        Self: Sized,
    {
        PredictIterator {
            iter: self,
            state: PredictState::new(vad),
        }
    }

    /// Creates a new [LabelIterator] from an iterator of samples.
    fn label<const N: usize>(
        self,
        vad: VoiceActivityDetector<N>,
        threshold: f32,
        padding_chunks: usize,
    ) -> LabelIterator<Self::Item, Self, N>
    where
        Self::Item: Sample,
        Self: Sized,
    {
        let state = LabelState::new(threshold, padding_chunks);
        LabelIterator {
            state,
            iter: self.predict(vad),
        }
    }
}

impl<I: Iterator> IteratorExt for I {}
