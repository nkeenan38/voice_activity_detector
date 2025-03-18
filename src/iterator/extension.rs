use crate::label::LabelState;
use crate::predict::PredictState;
use crate::{LabelIterator, PredictIterator, Sample, VoiceActivityDetector};

/// Extensions for iterators.
pub trait IteratorExt: Iterator {
    /// Creates a new [PredictIterator] from an iterator of samples.
    fn predict<'a>(
        self,
        vad: &'a mut VoiceActivityDetector<'a>,
    ) -> PredictIterator<'a, Self::Item, Self>
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
    fn label<'a>(
        self,
        vad: &'a mut VoiceActivityDetector<'a>,
        threshold: f32,
        padding_chunks: usize,
    ) -> LabelIterator<'a, Self::Item, Self>
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
