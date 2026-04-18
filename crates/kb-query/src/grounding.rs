use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingMetrics {
    pub citation_precision: f64,
    pub citation_recall: f64,
    pub hallucination_rate: f64,
    pub uncertainty_appropriate: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingThresholds {
    pub min_citation_precision: f64,
    pub min_citation_recall: f64,
    pub max_hallucination_rate: f64,
    pub min_uncertainty_accuracy: f64,
}

impl GroundingThresholds {
    #[must_use]
    pub const fn ship() -> Self {
        Self {
            min_citation_precision: 0.80,
            min_citation_recall: 0.50,
            max_hallucination_rate: 0.10,
            min_uncertainty_accuracy: 0.80,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingReport {
    pub mean_citation_precision: f64,
    pub mean_citation_recall: f64,
    pub mean_hallucination_rate: f64,
    pub uncertainty_accuracy: f64,
    pub per_question: Vec<QuestionGrounding>,
    pub passes_ship_threshold: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionGrounding {
    pub id: String,
    pub metrics: GroundingMetrics,
    pub verdict: GroundingVerdict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroundingVerdict {
    Strong,
    Acceptable,
    Weak,
    Ungrounded,
}

impl std::fmt::Display for GroundingVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Strong => write!(f, "strong"),
            Self::Acceptable => write!(f, "acceptable"),
            Self::Weak => write!(f, "weak"),
            Self::Ungrounded => write!(f, "ungrounded"),
        }
    }
}

pub struct AnswerEvidence {
    pub valid_citations: usize,
    pub invalid_citations: usize,
    pub total_manifest_entries: usize,
    pub expected_sources_cited: usize,
    pub total_expected_sources: usize,
    pub has_uncertainty_banner: bool,
    pub should_have_uncertainty: bool,
}

#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn score_answer(evidence: &AnswerEvidence) -> GroundingMetrics {
    let total_cited = evidence.valid_citations + evidence.invalid_citations;
    let citation_precision = if total_cited > 0 {
        evidence.valid_citations as f64 / total_cited as f64
    } else if evidence.total_manifest_entries == 0 {
        1.0
    } else {
        0.0
    };

    let citation_recall = if evidence.total_expected_sources > 0 {
        evidence.expected_sources_cited as f64 / evidence.total_expected_sources as f64
    } else {
        1.0
    };

    let hallucination_rate = if total_cited > 0 {
        evidence.invalid_citations as f64 / total_cited as f64
    } else {
        0.0
    };

    let uncertainty_appropriate =
        evidence.has_uncertainty_banner == evidence.should_have_uncertainty;

    GroundingMetrics {
        citation_precision,
        citation_recall,
        hallucination_rate,
        uncertainty_appropriate,
    }
}

#[must_use]
pub fn classify(metrics: &GroundingMetrics) -> GroundingVerdict {
    if metrics.hallucination_rate > 0.25 {
        return GroundingVerdict::Ungrounded;
    }
    if metrics.citation_precision >= 0.90 && metrics.citation_recall >= 0.60 {
        return GroundingVerdict::Strong;
    }
    if metrics.citation_precision >= 0.70 && metrics.citation_recall >= 0.40 {
        return GroundingVerdict::Acceptable;
    }
    GroundingVerdict::Weak
}

#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn aggregate(questions: &[QuestionGrounding], thresholds: &GroundingThresholds) -> GroundingReport {
    let n = questions.len() as f64;
    if questions.is_empty() {
        return GroundingReport {
            mean_citation_precision: 0.0,
            mean_citation_recall: 0.0,
            mean_hallucination_rate: 0.0,
            uncertainty_accuracy: 0.0,
            per_question: Vec::new(),
            passes_ship_threshold: false,
        };
    }

    let mean_precision: f64 =
        questions.iter().map(|q| q.metrics.citation_precision).sum::<f64>() / n;
    let mean_recall: f64 =
        questions.iter().map(|q| q.metrics.citation_recall).sum::<f64>() / n;
    let mean_hallucination: f64 =
        questions.iter().map(|q| q.metrics.hallucination_rate).sum::<f64>() / n;
    let uncertainty_correct = questions
        .iter()
        .filter(|q| q.metrics.uncertainty_appropriate)
        .count() as f64;
    let uncertainty_accuracy = uncertainty_correct / n;

    let passes = mean_precision >= thresholds.min_citation_precision
        && mean_recall >= thresholds.min_citation_recall
        && mean_hallucination <= thresholds.max_hallucination_rate
        && uncertainty_accuracy >= thresholds.min_uncertainty_accuracy;

    GroundingReport {
        mean_citation_precision: mean_precision,
        mean_citation_recall: mean_recall,
        mean_hallucination_rate: mean_hallucination,
        uncertainty_accuracy,
        per_question: questions.to_vec(),
        passes_ship_threshold: passes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_grounding() {
        let evidence = AnswerEvidence {
            valid_citations: 3,
            invalid_citations: 0,
            total_manifest_entries: 3,
            expected_sources_cited: 2,
            total_expected_sources: 2,
            has_uncertainty_banner: false,
            should_have_uncertainty: false,
        };
        let m = score_answer(&evidence);
        assert!((m.citation_precision - 1.0).abs() < f64::EPSILON);
        assert!((m.citation_recall - 1.0).abs() < f64::EPSILON);
        assert!(m.hallucination_rate.abs() < f64::EPSILON);
        assert!(m.uncertainty_appropriate);
        assert_eq!(classify(&m), GroundingVerdict::Strong);
    }

    #[test]
    fn hallucinated_citations() {
        let evidence = AnswerEvidence {
            valid_citations: 2,
            invalid_citations: 3,
            total_manifest_entries: 3,
            expected_sources_cited: 1,
            total_expected_sources: 2,
            has_uncertainty_banner: false,
            should_have_uncertainty: true,
        };
        let m = score_answer(&evidence);
        assert!((m.hallucination_rate - 0.6).abs() < f64::EPSILON);
        assert!(!m.uncertainty_appropriate);
        assert_eq!(classify(&m), GroundingVerdict::Ungrounded);
    }

    #[test]
    fn no_citations_with_sources_available() {
        let evidence = AnswerEvidence {
            valid_citations: 0,
            invalid_citations: 0,
            total_manifest_entries: 3,
            expected_sources_cited: 0,
            total_expected_sources: 2,
            has_uncertainty_banner: true,
            should_have_uncertainty: true,
        };
        let m = score_answer(&evidence);
        assert!(m.citation_precision.abs() < f64::EPSILON);
        assert!(m.citation_recall.abs() < f64::EPSILON);
        assert!(m.uncertainty_appropriate);
        assert_eq!(classify(&m), GroundingVerdict::Weak);
    }

    #[test]
    fn no_citations_no_sources() {
        let evidence = AnswerEvidence {
            valid_citations: 0,
            invalid_citations: 0,
            total_manifest_entries: 0,
            expected_sources_cited: 0,
            total_expected_sources: 0,
            has_uncertainty_banner: true,
            should_have_uncertainty: true,
        };
        let m = score_answer(&evidence);
        assert!((m.citation_precision - 1.0).abs() < f64::EPSILON);
        assert!((m.citation_recall - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn acceptable_grounding() {
        let evidence = AnswerEvidence {
            valid_citations: 3,
            invalid_citations: 1,
            total_manifest_entries: 4,
            expected_sources_cited: 2,
            total_expected_sources: 4,
            has_uncertainty_banner: false,
            should_have_uncertainty: false,
        };
        let m = score_answer(&evidence);
        assert!((m.citation_precision - 0.75).abs() < f64::EPSILON);
        assert!((m.citation_recall - 0.50).abs() < f64::EPSILON);
        assert_eq!(classify(&m), GroundingVerdict::Acceptable);
    }

    #[test]
    fn aggregate_report_passes() {
        let questions = vec![
            QuestionGrounding {
                id: "q1".to_string(),
                metrics: GroundingMetrics {
                    citation_precision: 1.0,
                    citation_recall: 1.0,
                    hallucination_rate: 0.0,
                    uncertainty_appropriate: true,
                },
                verdict: GroundingVerdict::Strong,
            },
            QuestionGrounding {
                id: "q2".to_string(),
                metrics: GroundingMetrics {
                    citation_precision: 0.8,
                    citation_recall: 0.6,
                    hallucination_rate: 0.05,
                    uncertainty_appropriate: true,
                },
                verdict: GroundingVerdict::Acceptable,
            },
        ];
        let report = aggregate(&questions, &GroundingThresholds::ship());
        assert!(report.passes_ship_threshold);
        assert!((report.mean_citation_precision - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn aggregate_report_fails_hallucination() {
        let questions = vec![QuestionGrounding {
            id: "q1".to_string(),
            metrics: GroundingMetrics {
                citation_precision: 0.9,
                citation_recall: 0.8,
                hallucination_rate: 0.15,
                uncertainty_appropriate: true,
            },
            verdict: GroundingVerdict::Acceptable,
        }];
        let report = aggregate(&questions, &GroundingThresholds::ship());
        assert!(!report.passes_ship_threshold);
    }

    #[test]
    fn ship_thresholds_are_sane() {
        let t = GroundingThresholds::ship();
        assert!(t.min_citation_precision > 0.5);
        assert!(t.min_citation_recall > 0.0);
        assert!(t.max_hallucination_rate < 0.5);
        assert!(t.min_uncertainty_accuracy > 0.5);
    }
}
