from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class FitQuality(str, Enum):
    GOOD = "good"
    MIXED = "mixed"
    POOR = "poor"
    INCONCLUSIVE = "inconclusive"  # corner plot ambiguous / insufficient


class ComplexityVerdict(str, Enum):
    TOO_SIMPLE = "too_simple"
    APPROPRIATE = "appropriate"
    TOO_COMPLEX = "too_complex"
    INCONCLUSIVE = "inconclusive"


class PosteriorConstraint(str, Enum):
    WELL_CONSTRAINED = "well_constrained"  # noticeably narrower than prior
    WEAKLY_CONSTRAINED = "weakly_constrained"  # only mild narrowing
    PRIOR_DOMINATED = "prior_dominated"  # looks like prior / flat
    BOUNDARY_PUSHED = "boundary_pushed"  # piles up against prior edge


class GeometryIssueType(str, Enum):
    NONE = "none"
    STRONG_CORRELATION = "strong_correlation"
    RIDGE_SHEET = "ridge_or_sheet"  # near-singular direction(s)
    FUNNEL = "funnel"
    CURVED_BANANA = "curved_banana"
    MULTI_MODAL = "multi_modal"
    SYMMETRY_LABEL_SWITCHING = "symmetry_or_label_switching"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RecommendedAction(str, Enum):
    REPARAMETERISE = "reparameterise"
    TIGHTEN_PRIORS = "tighten_priors"
    RELAX_PRIORS = "relax_priors"
    ADD_STRUCTURE = "add_structure"  # e.g., mixture/latent/hierarchical
    REMOVE_REDUNDANT_PARAMS = "remove_redundant_params"
    CHECK_SAMPLER = "check_sampler"
    RUN_PPC = "run_posterior_predictive_checks"
    COMPARE_MODELS = "compare_models"  # e.g., LOO/WAIC/Bayes factor
    COLLECT_MORE_DATA = "collect_more_data"
    OTHER = "other"


class Issue(BaseModel):
    issue_type: GeometryIssueType
    severity: Severity
    confidence: Confidence
    evidence: str = Field(
        ...,
        description="What in the corner plot supports this issue (e.g., 'marginal is flat across prior', 'two islands in theta1-theta2')."
    )
    likely_implication: str = Field(
        ...,
        description="Interpretation: identifiability, misspecification, symmetry, redundancy, etc."
    )
    recommended_actions: List[RecommendedAction] = Field(default_factory=list)
    action_notes: Optional[str] = Field(
        default=None,
        description="Optional details on how to execute the recommended actions."
    )


class ParameterAssessment(BaseModel):
    name: str
    constraint: PosteriorConstraint
    notes: Optional[str] = None


class Result(BaseModel):
    """
    Structured corner-plot diagnostics for a Bayesian imaging fit.
    """
    fit_quality: FitQuality
    model_complexity: ComplexityVerdict

    # Primary flags (kept for backward compatibility with your earlier interface)
    is_good_fit: bool
    may_be_multi_modal: bool

    # More detailed diagnostics
    overall_confidence: Confidence = Confidence.MEDIUM
    summary: str = Field(..., description="Concise executive summary of what the corner plot indicates.")
    key_issues: List[Issue] = Field(default_factory=list)

    parameter_assessments: List[ParameterAssessment] = Field(
        default_factory=list,
        description="Optional per-parameter posterior constraint status if parameter names are visible."
    )

    # Optional guidance (useful for automated follow-ups)
    top_next_steps: List[RecommendedAction] = Field(default_factory=list)
