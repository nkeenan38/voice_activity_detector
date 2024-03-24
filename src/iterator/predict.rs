use crate::predict::PredictState;
use crate::Sample;

/// Predicts speech in an iterator of audio samples.
pub struct PredictIterator<T, I, const N: usize>
where
    I: Iterator,
{
    pub(super) iter: I,
    pub(super) state: PredictState<T, N>,
}

impl<T, I, const N: usize> Iterator for PredictIterator<T, I, N>
where
    T: Sample,
    I: Iterator<Item = T>,
{
    type Item = ([T; N], f32);

    fn next(&mut self) -> Option<Self::Item> {
        for sample in self.iter.by_ref() {
            if let Some(value) = self.state.try_next(sample) {
                return Some(value);
            }
        }

        None
    }
}
