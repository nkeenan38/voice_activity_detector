use tokio_stream::StreamExt;
use voice_activity_detector::{StreamExt as _, VoiceActivityDetector};

#[tokio::test]
async fn wave_file_label_iterator() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("tests/.outputs")?;

    let mut reader = hound::WavReader::open("tests/samples/sample.wav")?;
    let spec = reader.spec();

    let mut speech = hound::WavWriter::create("tests/.outputs/label.stream.speech.wav", spec)?;
    let mut nonspeech =
        hound::WavWriter::create("tests/.outputs/label.stream.nonspeech.wav", spec)?;

    let vad = VoiceActivityDetector::builder()
        .chunk_size(256usize)
        .sample_rate(spec.sample_rate)
        .build()?;

    let chunks = reader.samples::<i16>().map_while(Result::ok);

    let mut chunks = tokio_stream::iter(chunks).label(vad, 0.5, 10);

    while let Some(chunk) = chunks.next().await {
        if chunk.is_speech() {
            for sample in chunk {
                speech.write_sample(sample)?;
            }
        } else {
            for sample in chunk {
                nonspeech.write_sample(sample)?;
            }
        }
    }

    speech.finalize()?;
    nonspeech.finalize()?;

    Ok(())
}
