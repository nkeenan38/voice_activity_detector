use crate::{Sample, VoiceActivityDetector};

pub struct PredictState<T, const N: usize> {
    vad: VoiceActivityDetector<N>,
    buffer: [T; N],
    buffer_size: usize,
}

impl<T, const N: usize> PredictState<T, N>
where
    T: Sample,
{
    pub fn new(vad: VoiceActivityDetector<N>) -> Self {
        Self {
            vad,
            buffer: [T::default(); N],
            buffer_size: 0,
        }
    }

    pub fn try_next(&mut self, sample: T) -> Option<([T; N], f32)> {
        self.buffer[self.buffer_size] = sample;
        self.buffer_size += 1;

        if self.buffer_size < N {
            return None;
        }

        let probability = self.vad.predict_array(self.buffer);
        let output = Some((self.buffer, probability));

        self.buffer.fill(T::default());
        self.buffer_size = 0;

        output
    }
}
