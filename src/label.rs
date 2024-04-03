use std::{collections::VecDeque, usize};

/// Labels a chunk of audio as either speech or non-speech.
#[derive(Clone, Debug)]
pub enum LabeledAudio<T> {
    /// The voice activity detector predicted a speech probability higher
    /// than the provided threshold.
    Speech(Vec<T>),
    /// The voice activity detector predicted a speech probability lower
    /// than the provided threshold.
    NonSpeech(Vec<T>),
}

impl<T> LabeledAudio<T> {
    /// Returns true if the audio label is AudioLabel::Speech
    pub fn is_speech(&self) -> bool {
        match &self {
            LabeledAudio::Speech(_) => true,
            LabeledAudio::NonSpeech(_) => false,
        }
    }

    /// Returns an iterator over the audio chunk slice.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        match &self {
            LabeledAudio::Speech(audio) => audio.iter(),
            LabeledAudio::NonSpeech(audio) => audio.iter(),
        }
    }
}

impl<T> IntoIterator for LabeledAudio<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            LabeledAudio::Speech(chunk) => chunk.into_iter(),
            LabeledAudio::NonSpeech(chunk) => chunk.into_iter(),
        }
    }
}

#[derive(Debug)]
enum LabelStateInner {
    /// Waiting for speech to start.
    Idle,
    /// Speech has started, flushes the buffer labeled
    FlushStartPadding,
    Active {
        speech: bool,
    }, // Speech until enough silence chunks
    FlushEndPadding,
}

#[derive(Debug)]
pub(crate) struct LabelState<T> {
    threshold: f32,
    padding_chunks: usize,
    buffer: VecDeque<Vec<T>>,
    state: LabelStateInner,
}

impl<T> LabelState<T> {
    pub fn new(threshold: f32, padding_chunks: usize) -> Self {
        Self {
            threshold,
            padding_chunks,
            buffer: VecDeque::with_capacity(padding_chunks + 1),
            state: LabelStateInner::Idle,
        }
    }

    pub fn try_buffer(&mut self) -> Option<LabeledAudio<T>> {
        match self.state {
            LabelStateInner::Idle => {
                // If the buffer has grown too large, return the oldest chunk as non-speech.
                if self.buffer.len() > self.padding_chunks {
                    if let Some(chunk) = self.buffer.pop_front() {
                        return Some(LabeledAudio::NonSpeech(chunk));
                    }
                }

                None
            }
            LabelStateInner::FlushStartPadding => {
                // Return any elements still in the buffer
                if let Some(chunk) = self.buffer.pop_front() {
                    return Some(LabeledAudio::Speech(chunk));
                }

                // If the buffer is drained, update the state to active
                self.state = LabelStateInner::Active { speech: false };
                None
            }
            LabelStateInner::Active { speech } => {
                if speech {
                    if let Some(chunk) = self.buffer.pop_front() {
                        return Some(LabeledAudio::Speech(chunk));
                    }
                }
                None
            }
            LabelStateInner::FlushEndPadding => {
                // Return any elements still in the buffer
                if let Some(chunk) = self.buffer.pop_front() {
                    return Some(LabeledAudio::Speech(chunk));
                }

                // If the buffer is drained, update the state to idle
                self.state = LabelStateInner::Idle;
                None
            }
        }
    }

    pub fn try_next(&mut self, chunk: Vec<T>, probability: f32) -> Option<LabeledAudio<T>> {
        match self.state {
            LabelStateInner::Idle => {
                // Add the chunk to the buffer
                self.buffer.push_back(chunk);

                // If speech has been detected, flush the buffer
                if probability >= self.threshold {
                    self.state = LabelStateInner::FlushStartPadding;
                    return self
                        .buffer
                        .pop_front()
                        .map(|chunk| LabeledAudio::Speech(chunk));
                }

                // If speech has not yet been detected and the buffer is full,
                // yield the earliest chunk.
                if self.buffer.len() > self.padding_chunks {
                    return self
                        .buffer
                        .pop_front()
                        .map(|chunk| LabeledAudio::NonSpeech(chunk));
                }

                // Otherwise, we don't have enough information to make a decision
                None
            }
            LabelStateInner::Active { ref mut speech } => {
                if probability >= self.threshold {
                    *speech = true;
                    if !self.buffer.is_empty() {
                        self.buffer.push_back(chunk);
                        self.buffer
                            .pop_front()
                            .map(|chunk| LabeledAudio::Speech(chunk))
                    } else {
                        Some(LabeledAudio::Speech(chunk))
                    }
                } else {
                    *speech = false;
                    self.buffer.push_back(chunk);
                    if self.buffer.len() >= self.padding_chunks {
                        self.state = LabelStateInner::FlushEndPadding;
                        self.buffer
                            .pop_front()
                            .map(|chunk| LabeledAudio::Speech(chunk))
                    } else {
                        None
                    }
                }
            }
            LabelStateInner::FlushStartPadding => None,
            LabelStateInner::FlushEndPadding => None,
        }
    }

    pub fn flush(&mut self) -> Option<LabeledAudio<T>> {
        match self.state {
            LabelStateInner::Idle => self
                .buffer
                .pop_front()
                .map(|chunk| LabeledAudio::NonSpeech(chunk)),
            _ => self
                .buffer
                .pop_front()
                .map(|chunk| LabeledAudio::Speech(chunk)),
        }
    }
}
