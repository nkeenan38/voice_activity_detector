# Voice Activity Detector

Provides a model and extensions for detecting speech in audio.

## Standalone Voice Activity Detector

This crate provides a standalone Voice Activity Detector (VAD) which can be used to predict speech in a chunk of audio. This implementation uses the [Silero VAD](https://github.com/snakers4/silero-vad).

The VAD predicts speech in a chunk of Linear Pulse Code Modulation (LPCM) encoded audio samples. These may be 8 or 16 bit integers or 32 bit floats.

The model is trained using chunk sizes of 256, 512, and 768 samples for an 8000 hz sample rate. It is trained using chunk sizes of 512, 768, 1024 samples for a 16,000 hz sample rate. These values are recommended for optimal performance, but are not required. The only requirement imposed by the underlying model is the sample rate must be no larger than 31.25 times the chunk size.

The samples passed to `predict` will be truncacted or padded if they are not of the correct length.

```rust
fn main() -> Result<(), voice_activity_detector::Error> {
    use voice_activity_detector::{VoiceActivityDetector};

    let chunk = vec![0i16; 512];
    let mut vad = VoiceActivityDetector::builder()
        .sample_rate(8000)
        .chunk_size(512usize)
        .build()?;
    let probability = vad.predict(chunk);
    println!("probability: {}", probability);

    Ok(())
}
```

## Extensions

Some extensions have been added for dealing with streams of audio. These extensions have variants to work with both Iterators and Async Iterators (Streams) of audio samples. The Stream utilities are enabled as part of the `async` feature.

### Predict Iterator/Stream

The PredictIterator and PredictStream work on an iterator/stream of samples, and return an iterator/stream containing a tuple of a chunk of audio and its probability of speech. Be sure to use the IteratorExt and StreamExt traits to bring the `predict` function on iterators into scope.

```rust
fn main() -> Result<(), voice_activity_detector::Error> {
    use voice_activity_detector::{IteratorExt, VoiceActivityDetector};

    let samples = [0i16; 5120];
    let mut vad = VoiceActivityDetector::builder()
        .sample_rate(8000)
        .chunk_size(512usize)
        .build()?;

    let probabilities = samples.into_iter().predict(&mut vad);
    for (chunk, probability) in probabilities {
        if probability > 0.5 {
            println!("speech detected!");
        }
    }
    Ok(())
}
```

### Label Iterator/Stream

The LabelIterator and LabelStream also work on an iterator/stream of samples. Rather than returning just the probability of speech for each chunk, these return labels of speech or non-speech. This helper allows adding additional padding to speech chunks to prevent sudden cutoffs of speech.

- `threshold`: Value between 0.0 and 1.0\. Probabilties greater than or equal to this value will be considered speech.
- `padding_chunks`: Adds additional chunks to the start and end of speech chunks.

```rust
fn main() -> Result<(), voice_activity_detector::Error> {
    use voice_activity_detector::{LabeledAudio, IteratorExt, VoiceActivityDetector};

    let samples = [0i16; 51200];
    let mut vad = VoiceActivityDetector::builder()
        .sample_rate(8000)
        .chunk_size(512usize)
        .build()?;

    // This will label any audio chunks with a probability greater than 75% as speech,
    // and label the 3 additional chunks before and after these chunks as speech.
    let labels = samples.into_iter().label(&mut vad, 0.75, 3);
    for label in labels {
        match label {
            LabeledAudio::Speech(_) => println!("speech detected!"),
            LabeledAudio::NonSpeech(_) => println!("non-speech detected!"),
        }
    }
    Ok(())
}
```

## Feature Flags

- `async`: Enables the structs and functions to work with `::future::Stream`.
- `load-dynamic`: By default, this library downloads prebuilt ONNX Runtime from Microsoft. This is convenient and works out of the box for most use cases. For the use cases that require more control, this feature flag enables the `load-dynamic` feature flag for the `ort` library. From the [ort library documentation](https://docs.rs/ort/latest/ort/#how-to-get-binaries):

> This doesn't link to any dynamic libraries, instead loading the libraries at runtime using dlopen(). This can be used to control the path to the ONNX Runtime binaries (meaning they don't always have to be directly next to your executable), and avoiding the shared library hell. To use this, enable the load-dynamic Cargo feature, and set the ORT_DYLIB_PATH environment variable to the path to your onnxruntime.dll/libonnxruntime.so/libonnxruntime.dylib - you can also use relative paths like ORT_DYLIB_PATH=./libonnxruntime.so (it will be relative to the executable). For convenience, you should download or compile ONNX Runtime binaries, put them in a permanent location, and set the environment variable permanently.

## More Examples

Please see the tests directory for more examples.

## Limitations

The voice activity detector and helper functions work only on mono-channel audio streams. If your use case involves multiple channels, you will need to split the channels and potentially interleave them again depending on your needs.

We have also currently not verified functionality with all platforms, here is what we tested: | Windows | macOS | Linux | | :-----: | :---: | :---: | | 🟢 | 🟢 | 🟢 |

🟢 = Available

🔵 = Currently in the works

🟡 = Currently not tested

🔴 = Not working currently (possible in the future)
