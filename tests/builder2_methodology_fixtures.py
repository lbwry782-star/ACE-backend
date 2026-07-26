"""
Shared Builder2 methodology test fixtures.
"""
from __future__ import annotations

from typing import Any, Dict

from engine.builder2_methodology_contract import METHODOLOGY_VERSION
from engine.builder2_strategy_identity import assign_strategy_foundation_identity, compute_strategy_foundation_digest


def methodology_strategy_extras(*, tournament_id: str = "test-tournament") -> Dict[str, Any]:
    base = {
        "methodologyVersion": METHODOLOGY_VERSION,
        "productNameResolved": "ACE Product",
        "language": "en",
        "problemPerception": {
            "statement": "Buyers struggle to see why this product beats familiar alternatives.",
            "groundingType": "common_market_behavior",
            "groundingEvidence": ["Customers compare against familiar agencies by default."],
            "whyItMatters": "The product must reframe the comparison.",
        },
        "relativeAdvantage": {
            "statement": "Closeness becomes the advantage.",
            "derivationFromProblem": "Because buyers default to familiar options, being closer is reframed as better fit.",
            "truthBoundary": "Does not claim universal superiority on every dimension.",
            "admitsRelevantGap": True,
        },
        "mechanismScan": {
            "domainFacts": ["People choose what feels nearest to their need."],
            "discoveredMechanism": "Physical closeness can express strategic closeness.",
            "creativeOpportunity": "Show closeness as the persuasive proof.",
            "depthEvidence": "The mechanism connects perceptual distance with strategic fit, not mere proximity cliché.",
        },
    }
    return assign_strategy_foundation_identity(base, tournament_id=tournament_id)


def methodology_strategy_identity_for(candidate_strategy: Dict[str, Any]) -> Dict[str, str]:
    return {
        "strategyFoundationId": str(candidate_strategy.get("strategyFoundationId") or ""),
        "strategyFoundationDigest": str(
            candidate_strategy.get("strategyFoundationDigest")
            or compute_strategy_foundation_digest(candidate_strategy)
        ),
    }


def _prototype_application(prototype_id: str) -> Dict[str, Any]:
    apps: Dict[str, Dict[str, Any]] = {
        "winning_card": {
            "winningCardApplication": {
                "mediumOrContainerIdentified": "The display surface of a storefront window",
                "whatItBecomes": "A proof of closeness to the customer",
                "whyTheTransformationProvesTheAdvantage": "The medium itself demonstrates strategic nearness.",
            }
        },
        "summer_fan": {
            "summerFanApplication": {
                "visibleBehavior": "A hand waves back and forth quickly",
                "inferredAbsentObject": "A cooling fan",
                "whyTheViewerInfersItWithoutExplanation": "The motion pattern matches familiar fan behavior.",
            }
        },
        "forgot": {
            "forgotApplication": {
                "omittedOrForgottenAction": "Turning on the light",
                "visibleConsequence": "A dark room remains dark while activity continues",
                "plannedContradiction": "Someone acts normally in darkness",
                "whyTheViewerSolvesIt": "The viewer infers the forgotten switch.",
            }
        },
        "greenpeace_essential_pairing": {
            "essentialPairingApplication": {
                "elementA": "A fragile natural element",
                "elementB": "Human responsibility",
                "essentialRelationship": "Protection follows from shared essence, not appearance",
                "notMerelyAppearance": "Shape similarity alone is rejected",
                "notMerelyFunction": "Utility alone is rejected",
                "notMerelyWordplay": "Wordplay alone is rejected",
                "emotionalRecognition": "The viewer feels the bond instantly.",
            }
        },
        "closest": {
            "closestApplication": {
                "admittedGap": "The brand is smaller than familiar agencies",
                "relativeNearness": "Strategic closeness to the client's need",
                "physicalOrVisualExpressionOfNearness": "Distance closes between two people",
                "whyThisIsHonestRatherThanInferior": "The admitted gap becomes the proof of fit.",
            }
        },
        "think_small": {
            "thinkSmallApplication": {
                "realWeakness": "The product footprint is physically small",
                "evidenceTheWeaknessIsReal": "Buyers notice limited size first",
                "acceptanceRatherThanDenial": "The ad accepts the small size openly",
                "reframing": "Small size becomes maneuverability",
                "relativeAdvantageCreated": "Agility becomes the strategic payoff",
            }
        },
    }
    return apps.get(prototype_id, {})


def realistic_core_candidate_extras(prototype_id: str, *, strategy: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Production-like Creator core without redundant analytical duplicate objects."""
    identity = methodology_strategy_identity_for(strategy or methodology_strategy_extras())
    app = _prototype_application(prototype_id)
    return {
        "methodologyVersion": METHODOLOGY_VERSION,
        **identity,
        "visualMechanism": "Closing distance makes strategic closeness visible through one human gesture.",
        "verbalPotential": {
            "decision": "not_needed",
            "reason": "The visible closing gesture communicates closeness without a headline.",
        },
        "runwayFeasibility": {
            "fitsSevenSeconds": True,
            "requiresImpossibleMorphing": False,
            "requiresSubtleUnseenInference": False,
        },
        "creatorReport": {
            "silentVerification": "The closing distance is visible without sound.",
            "puritySelfCheck": True,
        },
        **app,
    }


def methodology_candidate_extras(prototype_id: str, *, strategy: Dict[str, Any] | None = None) -> Dict[str, Any]:
    identity = methodology_strategy_identity_for(strategy or methodology_strategy_extras())
    return {
        "methodologyVersion": METHODOLOGY_VERSION,
        **identity,
        "visualMechanism": "Closing distance makes strategic closeness visible through one human gesture.",
        "creativeOrderConfirmation": {
            "visualCameBeforeKeyword": True,
            "runwayCheckCameBeforeKeyword": True,
            "headlineWasNotStartingPoint": True,
        },
        "prototypeMethodApplication": {
            "methodSummary": "Apply the reusable prototype method to the current problem.",
            "applicationToCurrentProblem": "The method reframes the buyer's comparison barrier.",
            "surfaceElementsCopied": [],
            "whyThisIsNotLiteralImitation": "Only the problem-solving approach is reused.",
        },
        "essenceExtreme": {
            "advantageEssence": "Strategic closeness",
            "extremePhysicalExpression": "Maximum visible closing of distance",
            "whyChosenObjectsFollowFromTheEssence": "Human proximity is the most direct proof of closeness.",
        },
        "visualFamilyConsistency": {
            "familyDefinition": "Human closeness gestures in one coherent world",
            "recurringMotif": "Closing distance",
            "whyAllVariationsBelongTogether": "Every beat shows the same nearness mechanism.",
            "sideBySideFrameTest": "Frames share the same spatial logic.",
        },
        "participationMechanism": {
            "whoOrWhatParticipates": "Two people",
            "visibleAction": "One step closes the distance",
            "visibleCauseAndEffect": "Space shrinks and contact follows",
            "notMerelyAReadyMadeResult": True,
        },
        "visualAnchor": {
            "description": "The moment the distance closes.",
            "whyEssential": "It proves closeness visually.",
            "appearsBeforeOrDuringResolution": True,
        },
        "anchorPunchlineSeparation": {
            "anchor": "The closing distance",
            "resolutionOrPunchline": "The embrace that follows",
            "whyTheyAreNotTheSameThing": "The anchor proves the mechanism; the resolution completes it.",
        },
        "runwayFeasibility": {
            "mainSubject": "Two people",
            "mainAction": "One person steps forward and they hug",
            "location": "Simple neutral room",
            "openingFrame": "Two people with visible space between them",
            "continuityRisk": "low",
            "generationRisks": [],
            "whyRunwayShouldUnderstand": "Single continuous human action in one room.",
            "fitsSevenSeconds": True,
            "requiresImpossibleMorphing": False,
            "requiresSubtleUnseenInference": False,
        },
        "verbalPotential": {
            "keywordOrKeyPhrase": "closer",
            "visualMeaning": "Physical closing of distance",
            "strategicMeaning": "Strategic fit with the buyer's need",
            "bornFromVisibleMechanism": True,
            "headlineMayBeUnnecessary": False,
        },
        "sourceConcept": {"type": "native_builder2"},
        "creatorReport": {
            "creatorPuritySelfCheck": "No other candidates, scores, or tournament data were referenced.",
        },
        **_prototype_application(prototype_id),
    }


def methodology_judgment_extras() -> Dict[str, Any]:
    return {
        "methodologyVersion": METHODOLOGY_VERSION,
        "problemAdvantageAssessment": "The advantage directly answers the grounded buyer problem.",
        "mechanismDepthAssessment": "The mechanism is deeper than a first association.",
        "prototypeMethodAssessment": "The prototype method is applied without literal surface copying.",
        "visualMechanismAssessment": "The visual mechanism expresses the advantage silently.",
        "participationAssessment": "The viewer witnesses the mechanism occurring.",
        "visualFamilyAssessment": "The visual family remains coherent.",
        "silentMovieAssessment": "The idea works without dialogue or explanatory sound.",
        "verbalLayerAssessment": {
            "applicability": "available",
            "keywordBornFromVisual": True,
            "visualMeaningIsClear": True,
            "strategicMeaningIsClear": True,
            "twoMeaningsReinforceEachOther": True,
            "notes": "The provisional keyword follows the visible mechanism.",
        },
        "headlineNecessityAssessment": {
            "headlineNeeded": False,
            "visualWouldWorkWithoutHeadline": True,
            "notes": "The visual proof is sufficient without a headline.",
        },
    }


def methodology_winner_extras(
    *,
    headline_decision: str = "include",
    winning_candidate: Dict[str, Any] | None = None,
    strategy: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    strategy_obj = strategy or methodology_strategy_extras()
    candidate = winning_candidate or methodology_candidate_extras("closest", strategy=strategy_obj)
    preservation = {
        "strategyFoundationId": strategy_obj.get("strategyFoundationId"),
        "prototypeId": candidate.get("prototypeId"),
        "structureType": candidate.get("structureType"),
        "visualParallelType": candidate.get("visualParallelType"),
        "coreCreativeMechanism": candidate.get("coreCreativeMechanism"),
    }
    headline_form = "none" if headline_decision == "omit" else "direct"
    return {
        "methodologyVersion": METHODOLOGY_VERSION,
        "headlineDecision": {
            "decision": headline_decision,
            "reason": "The visual mechanism is strong enough to decide headline inclusion explicitly.",
        },
        "headlineForm": headline_form,
        "preservationReference": preservation,
        "winningCandidatePreservationSnapshot": preservation,
        "winnerPreservationCheck": {
            "problemPreserved": True,
            "relativeAdvantagePreserved": True,
            "mechanismPreserved": True,
            "prototypeMethodPreserved": True,
            "visualParallelPreserved": True,
            "structurePreserved": True,
            "editingOnlyStrengthens": True,
        },
    }
