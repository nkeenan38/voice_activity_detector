use ort::{session::Session, session::builder::GraphOptimizationLevel};

use crate::{error::Error, Sample};

/// A voice activity detector session.
#[derive(Debug)]
pub struct VoiceActivityDetector {
    chunk_size: usize,
    sample_rate: i64,
    session: Session,
    h: ndarray::Array3<f32>,
    c: ndarray::Array3<f32>,
}

/// The silero ONNX model as bytes.
const MODEL: &[u8] = include_bytes!("silero_vad.onnx");

impl VoiceActivityDetector {
    /// Create a new [VoiceActivityDetectorBuilder].
    pub fn builder() -> VoiceActivityDetectorBuilder {
        VoiceActivityDetectorConfig::builder()
    }

    /// Gets the chunks size
    pub(crate) fn chunk_size(&self) -> usize {
        self.chunk_size
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
        let mut input = ndarray::Array2::<f32>::zeros((1, self.chunk_size));
        for (i, sample) in samples.into_iter().take(self.chunk_size).enumerate() {
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
        let hn = outputs
            .get("hn")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap();
        let cn = outputs
            .get("cn")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap();

        self.h.assign(&hn.view());
        self.c.assign(&cn.view());

        // Get the probability of speech.
        let output = outputs
            .get("output")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap();
        let probability = output.view()[[0, 0]];

        probability
    }
}

/// The configuration for the [VoiceActivityDetector]. Used to create
/// a [VoiceActivityDetectorBuilder] that performs runtime validation on build.
#[derive(Debug, typed_builder::TypedBuilder)]
#[builder(
    builder_method(vis = ""),
    builder_type(name = VoiceActivityDetectorBuilder, vis = "pub"),
    build_method(into = Result<VoiceActivityDetector, Error>, vis = "pub"))
]
struct VoiceActivityDetectorConfig {
    #[builder(setter(into))]
    chunk_size: usize,
    #[builder(setter(into))]
    sample_rate: i64,
    #[builder(default, setter(strip_option))]
    session: Option<Session>,
}

impl From<VoiceActivityDetectorConfig> for Result<VoiceActivityDetector, Error> {
    fn from(value: VoiceActivityDetectorConfig) -> Self {
        if (value.sample_rate as f32) / (value.chunk_size as f32) > 31.25 {
            return Err(Error::VadConfigError {
                sample_rate: value.sample_rate,
                chunk_size: value.chunk_size,
            });
        }

        let session = value.session.unwrap_or_else(|| {
            Session::builder()
                .unwrap()
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .unwrap()
                .with_intra_threads(1)
                .unwrap()
                .with_inter_threads(1)
                .unwrap()
                .commit_from_memory(MODEL)
                .unwrap()
        });

        Ok(VoiceActivityDetector {
            session,
            chunk_size: value.chunk_size,
            sample_rate: value.sample_rate,
            h: ndarray::Array3::<f32>::zeros((2, 1, 64)),
            c: ndarray::Array3::<f32>::zeros((2, 1, 64)),
        })
    }
}
