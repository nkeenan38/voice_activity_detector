use std::mem;

use crate::{Sample, VoiceActivityDetector};

pub struct PredictState<T> {
    vad: VoiceActivityDetector,
    buffer: Vec<T>,
}

impl<T> PredictState<T>
where
    T: Sample,
{
    pub fn new(vad: VoiceActivityDetector) -> Self {
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
