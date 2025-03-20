use crate::Sample;
use std::iter::Iterator;

use super::PredictIterator;

#[derive(Debug)]
pub(crate) struct SegmentMergerState<T> {
    threshold: f32,
    current_segment: Vec<T>,
    max_samples: usize,
    min_sil_chunks: usize,
    consecutive_silence: usize,
}

impl<T> SegmentMergerState<T> {
    pub fn new(threshold: f32, max_samples: usize, min_sil_chunks: usize) -> Self {
        Self {
            threshold,
            current_segment: Vec::new(),
            max_samples,
            min_sil_chunks,
            consecutive_silence: 0,
        }
    }
}

pub struct SegmentMergerIterator<'a, T, I>
where
    I: Iterator<Item = T>,
{
    pub(super) iter: PredictIterator<'a, T, I>,
    pub(super) state: SegmentMergerState<T>,
}

impl<T, I> Iterator for SegmentMergerIterator<'_, T, I>
where
    T: Sample,
    I: Iterator<Item = T>,
{
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let threshold = self.state.threshold;
        let max_samples = self.state.max_samples;
        let min_sil_chunks = self.state.min_sil_chunks;
        let mut current_segment = std::mem::take(&mut self.state.current_segment);
        let mut consecutive_silence = self.state.consecutive_silence;

        for (chunk, probability) in self.iter.by_ref() {
            if probability > threshold {
                if !current_segment.is_empty() && current_segment.len() + chunk.len() > max_samples
                {
                    self.state.current_segment = chunk;
                    self.state.consecutive_silence = 0;
                    return Some(current_segment);
                } else {
                    current_segment.extend(chunk);
                    consecutive_silence = 0;
                }
            } else {
                consecutive_silence += 1;
                if consecutive_silence >= min_sil_chunks && !current_segment.is_empty() {
                    self.state.current_segment = Vec::new();
                    self.state.consecutive_silence = consecutive_silence;
                    return Some(current_segment);
                }
            }
        }

        if !current_segment.is_empty() {
            return Some(current_segment);
        }
        None
    }
}
