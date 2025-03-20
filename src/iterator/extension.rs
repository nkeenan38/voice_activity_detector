use crate::label::LabelState;
use crate::predict::PredictState;
use crate::{LabelIterator, PredictIterator, Sample, VoiceActivityDetector};

use super::segment::{SegmentMergerIterator, SegmentMergerState};

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

    /// Creates a new [SegmentMergerIterator] from an iterator of samples.
    fn segment(
        self,
        vad: &mut VoiceActivityDetector,
        threshold: f32,
        max_speech_ms: usize,
        min_sil_ms: usize,
    ) -> SegmentMergerIterator<'_, Self::Item, Self>
    where
        Self::Item: Sample,
        Self: Sized,
    {
        let chunk_duration_ms = (vad.chunk_size as f64 / vad.sample_rate as f64) * 1000.0;
        let max_chunks = (max_speech_ms as f64 / chunk_duration_ms).ceil() as usize;
        let min_sil_chunks = (min_sil_ms as f64 / chunk_duration_ms).ceil() as usize;
        
        let max_samples = max_chunks * vad.chunk_size ;
        let state = SegmentMergerState::new(threshold, max_samples, min_sil_chunks);
        SegmentMergerIterator {
            iter: self.predict(vad),
            state,
        }
    }
}

impl<I: Iterator> IteratorExt for I {}
