//! Device calls work from any thread, including after the thread that made the
//! process's first device call has exited.
//!
//! Each case runs in a child process: this test binary, run again on this test
//! with the case named in `DECIBRI_DEVICE_THREADS_CASE`. The case's first
//! device call is therefore the first in its process, and a fault ends the
//! child rather than the test run. The parent requires every child to exit
//! normally and to print the case's completion line, so a child that ran no
//! test fails as well.
//!
//! The cases need no audio device. With none present the calls return errors
//! or empty lists, which the cases ignore.

#![cfg(any(feature = "capture", feature = "playback"))]

use std::process::Command;
use std::thread;

/// Names the case a child process runs.
const CASE_VAR: &str = "DECIBRI_DEVICE_THREADS_CASE";

/// This test's name, as the test harness matches it.
const TEST_NAME: &str = "device_calls_work_after_the_first_device_thread_exits";

/// Lists the input and output devices, ignoring the result.
fn list_devices() {
    let _ = decibri::input_devices();
    let _ = decibri::output_devices();
}

/// The cases this build supports, named for what the first thread does.
fn cases() -> Vec<&'static str> {
    [
        Some("list"),
        cfg!(feature = "capture").then_some("microphone"),
        cfg!(feature = "playback").then_some("speaker"),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// Runs one case. A thread makes the process's first device call and exits,
/// then a second thread lists the devices and exits, then this thread lists
/// them.
fn run_case(case: &str) {
    let first: fn() = match case {
        "list" => list_devices,
        #[cfg(feature = "capture")]
        "microphone" => || {
            let _ = decibri::Microphone::new(decibri::MicrophoneConfig::default());
        },
        #[cfg(feature = "playback")]
        "speaker" => || {
            let _ = decibri::Speaker::new(decibri::SpeakerConfig::default());
        },
        other => panic!("unknown case {other}"),
    };
    thread::spawn(first)
        .join()
        .expect("the first device thread completes");
    thread::spawn(list_devices)
        .join()
        .expect("the second device thread completes");
    list_devices();
    println!("case {case} complete");
}

#[test]
fn device_calls_work_after_the_first_device_thread_exits() {
    if let Ok(case) = std::env::var(CASE_VAR) {
        run_case(&case);
        return;
    }
    let binary = std::env::current_exe().expect("the test binary's path");
    let mut failures = Vec::new();
    for case in cases() {
        let output = Command::new(&binary)
            .args([TEST_NAME, "--exact", "--nocapture", "--test-threads=1"])
            .env(CASE_VAR, case)
            .output()
            .expect("the child process starts");
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !output.status.success() || !stdout.contains(&format!("case {case} complete")) {
            failures.push(format!(
                "case {case}: the child ended with {}\nstdout:\n{stdout}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
