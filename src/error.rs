/// An enum of all errors returned by the voice activity detector functions.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// The VAD configuration must use a smaller chunk size or sample rate.
    #[error("the chunk size {chunk_size} is too small for the provided sample rate {sample_rate}")]
    VadConfigError {
        /// The sample rate for the VAD.
        sample_rate: i64,
        /// The chunk size for the VAD.
        chunk_size: usize,
    },
}
