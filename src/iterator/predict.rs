use crate::predict::PredictState;
use crate::Sample;

/// Predicts speech in an iterator of audio samples.
pub struct PredictIterator<'a, T, I>
where
    I: Iterator,
{
    pub(super) iter: I,
    pub(super) state: PredictState<'a, T>,
}

impl<T, I> Iterator for PredictIterator<'_, T, I>
where
    T: Sample,
    I: Iterator<Item = T>,
{
    type Item = (Vec<T>, f32);

    fn next(&mut self) -> Option<Self::Item> {
        for sample in self.iter.by_ref() {
            if let Some(value) = self.state.try_next(sample) {
                return Some(value);
            }
        }

        None
    }
}
