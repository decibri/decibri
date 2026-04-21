#[cfg(all(feature = "ort-load-dynamic", feature = "ort-download-binaries"))]
compile_error!(
    "features `ort-load-dynamic` and `ort-download-binaries` are mutually exclusive; \
     select exactly one. The default feature set enables `ort-load-dynamic`; \
     if you want `ort-download-binaries`, disable default features first."
);

pub mod error;
pub mod sample;

#[cfg(any(feature = "capture", feature = "output"))]
pub mod device;

#[cfg(feature = "capture")]
pub mod capture;

#[cfg(feature = "output")]
pub mod output;

#[cfg(feature = "vad")]
pub mod vad;

#[cfg(feature = "denoise")]
pub mod denoise;

#[cfg(feature = "gain")]
pub mod gain;
