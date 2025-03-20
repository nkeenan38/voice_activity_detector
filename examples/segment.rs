use hound::WavReader;

fn main() -> Result<(), voice_activity_detector::Error> {
    use voice_activity_detector::{IteratorExt, VoiceActivityDetector};

    let mut wav_reader = WavReader::open("/home/kautism/.cache/huggingface/hub/models--happyme531--SenseVoiceSmall-RKNN2/snapshots/01bc98205905753b7caafd6da25c84fba2490b59/output.wav").expect("");

    let content = wav_reader
        .samples()
        .filter_map(|x| x.ok())
        .collect::<Vec<i16>>();
    let mut vad = VoiceActivityDetector::builder()
        .sample_rate(16000)
        .chunk_size(512usize)
        .build()?;

    // This will label any audio chunks with a probability greater than 75% as speech,
    let segments = content.into_iter().segment(&mut vad, 0.75, 9000, 300);
    for segment in segments {
        println!("{}", segment.len());
    }
    Ok(())
}
