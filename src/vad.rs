use ort::{GraphOptimizationLevel, Session};

use crate::{error::Error, Sample};

/// A voice activity detector session.
#[derive(Debug)]
pub struct VoiceActivityDetector<const N: usize> {
    sample_rate: i64,
    session: ort::Session,
    h: ndarray::Array3<f32>,
    c: ndarray::Array3<f32>,
}

/// The silero ONNX model as bytes.
const MODEL: &[u8] = include_bytes!("silero_vad.onnx");

impl<const N: usize> VoiceActivityDetector<N> {
    /// Creates a new [VoiceActivityDetector].
    pub fn try_with_sample_rate(sample_rate: impl Into<i64>) -> Result<Self, Error> {
        let sample_rate: i64 = sample_rate.into();
        if (sample_rate as f32) / (N as f32) > 31.25 {
            return Err(Error::VadConfigError {
                sample_rate,
                chunk_size: N,
            });
        }

        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(1)
            .unwrap()
            .with_inter_threads(1)
            .unwrap()
            .with_model_from_memory(MODEL)
            .unwrap();

        Ok(Self::with_session(session, sample_rate))
    }

    /// Creates a new [VoiceActivityDetector] using the provided ONNX runtime session.
    ///
    /// Use this if the default ONNX session configuration is not to your liking.
    pub fn with_session(session: Session, sample_rate: impl Into<i64>) -> Self {
        Self {
            session,
            sample_rate: sample_rate.into(),
            h: ndarray::Array3::<f32>::zeros((2, 1, 64)),
            c: ndarray::Array3::<f32>::zeros((2, 1, 64)),
        }
    }

    /// Resets the state of the voice activity detector session.
    pub fn reset(&mut self) {
        self.h.fill(0f32);
        self.c.fill(0f32);
    }

    /// Predicts the existence of speech in a single iterable of audio.
    ///
    /// The samples iterator will be padded if it is too short, or truncated if it is
    /// too long.
    pub fn predict<S, I>(&mut self, samples: I) -> f32
    where
        S: Sample,
        I: IntoIterator<Item = S>,
    {
        let mut input = ndarray::Array2::<f32>::zeros((1, N));
        for (i, sample) in samples.into_iter().take(N).enumerate() {
            input[[0, i]] = sample.to_f32();
        }

        let sample_rate = ndarray::arr1::<i64>(&[self.sample_rate]);

        let inputs = ort::inputs![
            "input" => input.view(),
            "sr" => sample_rate.view(),
            "h" => self.h.view(),
            "c" => self.c.view(),
        ]
        .unwrap();

        let outputs = self.session.run(inputs).unwrap();

        // Update h and c recursively.
        let hn = outputs.get("hn").unwrap().extract_tensor::<f32>().unwrap();
        let cn = outputs.get("cn").unwrap().extract_tensor::<f32>().unwrap();

        self.h.assign(&hn.view());
        self.c.assign(&cn.view());

        // Get the probability of speech.
        let output = outputs
            .get("output")
            .unwrap()
            .extract_tensor::<f32>()
            .unwrap();
        let probability = output.view()[[0, 0]];

        probability
    }

    /// Predicts the existence of speech in an array of audio samples.
    ///
    /// This is provided as an alternative to [Self::predict] in order to
    /// guarantee the samples are the exact correct length.
    pub fn predict_array<S>(&mut self, samples: [S; N]) -> f32
    where
        S: Sample,
    {
        self.predict(samples)
    }
}
