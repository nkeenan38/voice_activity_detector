use std::task::Poll;

use futures::Stream;
use pin_project::pin_project;

use crate::{
    utilities::{LabelState, LabeledAudio, PredictState},
    Error, Sample, VoiceActivityDetector,
};

/// Predicts speech in a stream of audio samples.
#[pin_project]
pub struct PredictStream<T, St, const N: usize>
where
    St: Stream,
{
    #[pin]
    stream: St,
    state: PredictState<T, N>,
}

impl<T, St, const N: usize> Stream for PredictStream<T, St, N>
where
    T: Sample,
    St: Stream<Item = T>,
{
    type Item = Result<([T; N], f32), Error>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        loop {
            let sample = match this.stream.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Ready(Some(next)) => next,
            };
            match this.state.try_next(sample) {
                None => continue,
                Some(value) => return Poll::Ready(Some(value)),
            }
        }
    }
}

/// Labels an iterator of speech samples as either speech or non-speech according
/// to the provided speech sensitity.
#[pin_project]
pub struct LabelStream<T, S, const N: usize>
where
    S: Stream,
{
    #[pin]
    stream: PredictStream<T, S, N>,
    state: LabelState<T, N>,
}

impl<T, S, const N: usize> Stream for LabelStream<T, S, N>
where
    T: Sample,
    S: Stream<Item = T>,
{
    type Item = Result<LabeledAudio<T, N>, Error>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // If there is any buffered audio ready, return it.
        if let Some(buffered) = this.state.try_buffer() {
            return Poll::Ready(Some(Ok(buffered)));
        }

        loop {
            // Try to get the next chunk of audio. If there is no audio remaining, flush any
            // buffered audio.
            let (chunk, probability) = match this.stream.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Err(error))) => return Poll::Ready(Some(Err(error))),
                Poll::Ready(None) => return Poll::Ready(this.state.flush().map(Ok)),
                Poll::Ready(Some(Ok(value))) => value,
            };

            // If this audio chunk has resulted in a definitive result, return in.
            if let Some(audio) = this.state.try_next(chunk, probability) {
                return Poll::Ready(Some(Ok(audio)));
            }
        }
    }
}

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
        min_silence_chunks: usize,
        padding_chunks: usize,
    ) -> LabelStream<Self::Item, Self, N>
    where
        Self::Item: Sample,
        Self: Sized,
    {
        let state = LabelState::new(threshold, min_silence_chunks, padding_chunks);
        LabelStream {
            state,
            stream: self.predict(vad),
        }
    }
}

impl<I: Stream> StreamExt for I {}
