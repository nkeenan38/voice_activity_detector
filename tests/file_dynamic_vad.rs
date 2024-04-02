use std::error::Error;

use itertools::Itertools;
use voice_activity_detector::DynamicVoiceActivityDetector;

#[test]
fn wave_file_dynamic_detector() -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all("tests/.outputs")?;

    let mut reader = hound::WavReader::open("tests/samples/sample.wav")?;
    let spec = reader.spec();

    let mut speech = hound::WavWriter::create("tests/.outputs/dynamic.iter.speech.wav", spec)?;
    let mut nonspeech =
        hound::WavWriter::create("tests/.outputs/dynamic.iter.nonspeech.wav", spec)?;

    let mut vad = DynamicVoiceActivityDetector::try_with_sample_rate(256usize, spec.sample_rate)?;

    let samples = reader
        .samples::<i16>()
        .map(|s| s.unwrap())
        .collect::<Vec<_>>();
    let chunks = samples
        .iter()
        .chunks(256)
        .into_iter()
        .map(|chunk| chunk.collect::<Vec<_>>())
        .collect::<Vec<_>>();

    for chunk in chunks {
        let ready_chunk = chunk.iter().map(|s| **s).collect::<Vec<_>>();
        let probability = vad.predict(ready_chunk);
        if probability > 0.5 {
            for sample in chunk {
                speech.write_sample(*sample)?;
            }
        } else {
            for sample in chunk {
                nonspeech.write_sample(*sample)?;
            }
        }
    }

    speech.finalize()?;
    nonspeech.finalize()?;

    Ok(())
}
