use futures::Stream;

use crate::label::LabelState;
use crate::predict::PredictState;
use crate::{LabelStream, PredictStream, Sample, VoiceActivityDetector};

/// Extensions for streams.
pub trait StreamExt: Stream {
    /// Creates a new [PredictStream] from a stream of samples.
    fn predict<const N: usize>(
        self,
        vad: VoiceActivityDetector<N>,
    ) -> PredictStream<Self::Item, Self, N>
    where
        Self::Item: Sample,
        Self: Sized,
    {
        PredictStream {
            stream: self,
            state: PredictState::new(vad),
        }
    }

    /// Creates a new [LabelStream] from an iterator of samples.
    fn label<const N: usize>(
        self,
        vad: VoiceActivityDetector<N>,
        threshold: f32,
        padding_chunks: usize,
    ) -> LabelStream<Self::Item, Self, N>
    where
        Self::Item: Sample,
        Self: Sized,
    {
        let state = LabelState::new(threshold, padding_chunks);
        LabelStream {
            state,
            stream: self.predict(vad),
        }
    }
}

impl<I: Stream> StreamExt for I {}
