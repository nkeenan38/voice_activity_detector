use std::future;

use futures::{Stream, StreamExt};
use hound::WavSpec;
use voice_activity_detector::{LabeledAudio, StreamExt as _, VoiceActivityDetector};

/// Writes the stream to a file. Returns true if the stream is empty.
async fn write(
    mut stream: impl Stream<Item = LabeledAudio<i16>> + Unpin,
    iteration: usize,
    spec: WavSpec,
) -> Result<bool, Box<dyn std::error::Error>> {
    let filename = format!("tests/.outputs/chunk_stream.{iteration}.wav");
    let mut file = hound::WavWriter::create(filename, spec)?;

    let mut empty = true;
    while let Some(audio) = stream.next().await {
        empty = false;
        for sample in audio {
            file.write_sample(sample)?;
        }
    }

    file.finalize()?;
    Ok(empty)
}

#[tokio::test]
async fn chunk_stream() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("tests/.outputs")?;

    let mut reader = hound::WavReader::open("tests/samples/sample.wav")?;
    let spec = reader.spec();

    let mut vad = VoiceActivityDetector::builder()
        .sample_rate(8000)
        .chunk_size(512usize)
        .build()
        .unwrap();

    let chunks = reader.samples::<i16>().map_while(Result::ok);
    let mut labels = tokio_stream::iter(chunks).label(&mut vad, 0.75, 3).fuse();

    for i in 0.. {
        let next = labels
            .by_ref()
            .skip_while(|audio| future::ready(!audio.is_speech()))
            .take_while(|audio| future::ready(audio.is_speech()));

        let empty = write(next, i, spec).await?;
        if empty {
            break;
        }
    }

    Ok(())
}
