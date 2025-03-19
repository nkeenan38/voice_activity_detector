use crate::label::LabelState;
use crate::predict::PredictState;
use crate::{LabelIterator, PredictIterator, Sample, VoiceActivityDetector};

/// Extensions for iterators.
pub trait IteratorExt: Iterator {
    /// Creates a new [PredictIterator] from an iterator of samples.
    fn predict(self, vad: &mut VoiceActivityDetector) -> PredictIterator<'_, Self::Item, Self>
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
    fn label(
        self,
        vad: &mut VoiceActivityDetector,
        threshold: f32,
        padding_chunks: usize,
    ) -> LabelIterator<'_, Self::Item, Self>
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
