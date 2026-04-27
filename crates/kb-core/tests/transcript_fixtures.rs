//! Round-trip kbtx parsing against real LiveRamp transcripts.
//!
//! The fixtures are the rendered outputs from the bn-3hfw Python tool,
//! checked into the repo so the Rust parser/renderer is validated against
//! real content (not just the synthetic unit-test sample).

use kb_core::transcript;

const USB: &str = include_str!("fixtures/usb-team-intro.kbtx.md");
const DELTA: &str = include_str!("fixtures/delta-team-intro.kbtx.md");

#[test]
fn parses_usb_transcript() {
    let doc = transcript::parse(USB).expect("USB parses cleanly");
    assert!(doc.frontmatter.contains("USB Team Intro"));
    assert_eq!(doc.roster.len(), 6, "USB has 6 speakers in roster");
    let turns = doc.body.iter().filter(|n| matches!(n, transcript::BodyNode::Turn(_))).count();
    let actions = doc.body.iter().filter(|n| matches!(n, transcript::BodyNode::ActionLine(_))).count();
    assert_eq!(turns, 188, "USB has 188 speaker turns");
    assert_eq!(actions, 3, "USB has 3 action lines (synthetic pauses)");
}

#[test]
fn parses_delta_transcript() {
    let doc = transcript::parse(DELTA).expect("Delta parses cleanly");
    assert!(doc.frontmatter.contains("Delta Team Intro"));
    let turns = doc.body.iter().filter(|n| matches!(n, transcript::BodyNode::Turn(_))).count();
    let actions = doc.body.iter().filter(|n| matches!(n, transcript::BodyNode::ActionLine(_))).count();
    assert_eq!(turns, 106);
    assert_eq!(actions, 8);
}

#[test]
fn round_trip_usb_byte_identical() {
    let doc = transcript::parse(USB).unwrap();
    let rendered = transcript::render(&doc);
    assert_eq!(rendered, USB, "USB transcript must round-trip byte-identical");
}

#[test]
fn round_trip_delta_byte_identical() {
    let doc = transcript::parse(DELTA).unwrap();
    let rendered = transcript::render(&doc);
    assert_eq!(rendered, DELTA, "Delta transcript must round-trip byte-identical");
}
