use std::error::Error;

use voice_activity_detector::{IteratorExt, VoiceActivityDetector};

#[test]
fn wave_file_label_iterator() -> Result<(), Box<dyn Error>> {
    let mut reader = hound::WavReader::open("tests/samples/sample.wav")?;
    let spec = reader.spec();

    let mut speech = hound::WavWriter::create("tests/.outputs/label.iter.speech.wav", spec)?;
    let mut nonspeech = hound::WavWriter::create("tests/.outputs/label.iter.nonspeech.wav", spec)?;

    let vad = VoiceActivityDetector::<256>::try_with_sample_rate(spec.sample_rate)?;

    let chunks = reader
        .samples::<i16>()
        .map_while(Result::ok)
        .label(vad, 0.5, 10);

    for chunk in chunks {
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
