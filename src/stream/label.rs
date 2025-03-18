use std::task::Poll;

use futures::Stream;
use pin_project::pin_project;

use crate::label::{LabelState, LabeledAudio};
use crate::{PredictStream, Sample};

/// Labels a stream of speech samples as either speech or non-speech according
/// to the provided speech sensitity.
#[pin_project]
pub struct LabelStream<'a, T, St>
where
    St: Stream,
{
    #[pin]
    pub(super) stream: PredictStream<'a, T, St>,
    pub(super) state: LabelState<T>,
}

impl<T, St> Stream for LabelStream<'_, T, St>
where
    T: Sample,
    St: Stream<Item = T>,
{
    type Item = LabeledAudio<T>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if let Some(buffered) = this.state.try_buffer() {
            return Poll::Ready(Some(buffered));
        }

        loop {
            let next = this.stream.as_mut().poll_next(cx);
            let (chunk, probability) = match next {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => return Poll::Ready(this.state.flush()),
                Poll::Ready(Some(value)) => value,
            };

            if let Some(audio) = this.state.try_next(chunk, probability) {
                return Poll::Ready(Some(audio));
            }
        }
    }
}
