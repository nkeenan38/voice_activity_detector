use std::task::Poll;

use futures::Stream;
use pin_project::pin_project;

use crate::predict::PredictState;
use crate::Sample;

/// Predicts speech in a stream of audio samples.
#[pin_project]
pub struct PredictStream<T, St, const N: usize>
where
    St: Stream,
{
    #[pin]
    pub(super) stream: St,
    pub(super) state: PredictState<T, N>,
}

impl<T, St, const N: usize> Stream for PredictStream<T, St, N>
where
    T: Sample,
    St: Stream<Item = T>,
{
    type Item = ([T; N], f32);

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
