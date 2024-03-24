/// A single sample of Linear Pulse Code Modulation (LPCM) encoded audio.
///
/// Integers are between a range of -32768 to 32768.
/// Floats are between -1.0 and 1.0.
pub trait Sample: Copy + Default + Sized {
    /// Convert the sample to a float.
    fn to_f32(self) -> f32;
}

impl Sample for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}

impl Sample for i16 {
    fn to_f32(self) -> f32 {
        f32::from(self) / 32768.0
    }
}

impl Sample for i8 {
    fn to_f32(self) -> f32 {
        f32::from(self) / 32768.0
    }
}

impl Sample for u16 {
    fn to_f32(self) -> f32 {
        f32::from(self) / 32768.0
    }
}

impl Sample for u8 {
    fn to_f32(self) -> f32 {
        f32::from(self) / 32768.0
    }
}
