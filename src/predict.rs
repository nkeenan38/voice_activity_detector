use std::mem;

use crate::{Sample, VoiceActivityDetector};

pub struct PredictState<'a, T> {
    vad: &'a mut VoiceActivityDetector<'a>,
    buffer: Vec<T>,
}

impl<'a, T> PredictState<'a, T>
where
    T: Sample,
{
    pub fn new(vad: &'a mut VoiceActivityDetector<'a>) -> Self {
        let chunk_size = vad.chunk_size();
        Self {
            vad,
            buffer: Vec::with_capacity(chunk_size),
        }
    }

    pub fn try_next(&mut self, sample: T) -> Option<(Vec<T>, f32)> {
        self.buffer.push(sample);
        if self.buffer.len() < self.vad.chunk_size() {
            return None;
        }

        let probability = self.vad.predict(self.buffer.iter().copied());
        let buffer = mem::replace(&mut self.buffer, Vec::with_capacity(self.vad.chunk_size()));

        Some((buffer, probability))
    }
}
