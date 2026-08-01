"""
Shared Builder2 methodology test fixtures.
"""
from __future__ import annotations

from typing import Any, Dict

from engine.builder2_methodology_contract import METHODOLOGY_VERSION
from engine.builder2_strategy_identity import assign_strategy_foundation_identity, compute_strategy_foundation_digest
from engine.builder2_complete_ad_contract import (
    build_default_creator_advertising_closure,
    build_default_creator_semantic_bridge,
)
from engine.builder2_advertising_slogan_quality_contract import (
    BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION,
    CREATOR_SLOGAN_FORMULATION_KEY,
    WINNER_SLOGAN_EVIDENCE_KEY,
    build_default_creator_slogan_formulation,
    build_default_judge_slogan_assessment,
)


def advertising_slogan_quality_creator_extras(
    *,
    relative_advantage_source: str,
    final_slogan_text: str,
    transformation_type: str = "direct_distillation",
    why_advertising: str = "The line compresses the relative advantage into a memorable closing claim.",
) -> Dict[str, Any]:
    return {
        CREATOR_SLOGAN_FORMULATION_KEY: build_default_creator_slogan_formulation(
            relative_advantage_source=relative_advantage_source,
            final_slogan_text=final_slogan_text,
            transformation_type=transformation_type,
            why_advertising=why_advertising,
        ),
    }


def advertising_slogan_quality_judge_extras(*, notes: str = "") -> Dict[str, Any]:
    return {"advertisingSloganAssessment": build_default_judge_slogan_assessment(notes=notes)}


def advertising_slogan_quality_winner_extras(
    *,
    relative_advantage_source: str,
    final_slogan_text: str,
    transformation_type: str = "direct_distillation",
    why_advertising: str = "The winning Creator slogan remains the canonical advertising formulation.",
) -> Dict[str, Any]:
    return {
        WINNER_SLOGAN_EVIDENCE_KEY: build_default_creator_slogan_formulation(
            relative_advantage_source=relative_advantage_source,
            final_slogan_text=final_slogan_text,
            transformation_type=transformation_type,
            why_advertising=why_advertising,
        ),
    }


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


def complete_ad_creator_extras(
    *,
    product_name: str = "ACE Product",
    slogan_text: str = "קרוב יותר ממה שחשבת",
    language: str = "he",
    key_word: str = "closer",
    relative_advantage_source: str = "Closeness becomes the advantage.",
) -> Dict[str, Any]:
    return {
        "advertisingClosure": build_default_creator_advertising_closure(
            product_name=product_name,
            slogan_text=slogan_text,
            language=language,
        ),
        "semanticBridge": build_default_creator_semantic_bridge(
            key_word=key_word,
            visual_meaning="Physical closing of distance between two people",
            slogan_meaning="Strategic closeness to the buyer's need",
            strategic_meaning="Closeness becomes the advantage",
            how_they_meet="The visible gesture proves the same strategic promise the slogan closes",
        ),
        **advertising_slogan_quality_creator_extras(
            relative_advantage_source=relative_advantage_source,
            final_slogan_text=slogan_text,
        ),
    }


def logo_policy_creator_extras(*, advertised_entity_name: str = "ACE Product") -> Dict[str, Any]:
    return {
        "logoPolicyReport": {
            "advertisedEntityName": advertised_entity_name,
            "logoDependentConcept": False,
            "advertisedLogoRequested": False,
            "thirdPartyBrandingRisk": "None; scene uses generic unbranded objects only.",
            "inventedLogoRisk": "None; no invented mark or wordmark is planned.",
            "brandedObjectRisk": "None; packaging, clothing, screens, and vehicles remain unmarked.",
            "logoFreeSceneDescription": "Generic unbranded objects express the mechanism with no visible marks.",
            "genericObjectSubstitutions": "Any potentially branded object type is replaced with a neutral generic form.",
            "plainTextNameReservedForClosureOnly": True,
            "logoPolicySatisfied": True,
        }
    }


def logo_policy_judge_extras() -> Dict[str, Any]:
    return {
        "logoPolicyAssessment": {
            "logoDetectedInPlan": False,
            "logoDependentMeaning": False,
            "advertisedLogoRequested": False,
            "thirdPartyBrandingDetected": False,
            "inventedLogoDetected": False,
            "brandedObjectRiskAccepted": True,
            "plainTextIdentificationOnly": True,
            "logoFreeExecutionAccepted": True,
            "logoPolicySatisfied": True,
            "rejectionReason": None,
        }
    }


def metaphorical_embodiment_creator_extras(
    *,
    strategic_perception: str = "Physical closeness can express strategic fit better than generic scale.",
    creative_embodiment_mode: str = "transformed_action_or_motion",
    literal_symbol_disposition: str = "rejected",
) -> Dict[str, Any]:
    return {
        "metaphoricalEmbodiment": {
            "strategicPerception": strategic_perception,
            "obviousLiteralVisualSymbols": ["agency office proximity map", "location pin dashboard"],
            "literalSymbolDisposition": literal_symbol_disposition,
            "literalSymbolsRejectedOrTransformed": "Reject map pins; show closeness through human distance closing.",
            "creativeEmbodimentMode": creative_embodiment_mode,
            "embodimentSubjectOrWorld": "Two people in a neutral room",
            "physicalEmbodiment": "One person closes the visible distance to another",
            "embodiedStrategicRelationship": "Nearness in the frame expresses strategic closeness to the buyer's need",
            "visiblePhysicalRelationship": "The shrinking gap between two people is the main visible proof",
            "transformationMechanism": "Distance closing transforms a competitive gap into visible fit",
            "whyTheVisualIsNotLiteralExplanation": "Closeness is shown through bodies, not a location report",
            "understandableBeforeSlogan": True,
            "sloganBridgeToBusinessMeaning": "The slogan names the closeness the viewer already saw",
        },
        "visualBridgeAssessment": {
            "centralVisibleDetail": "The moment the distance between two people closes",
            "sloganConnectionToVisibleDetail": "The slogan completes the visible closing gesture",
            "sloganConnectionToRelativeAdvantage": "The slogan closes the relative advantage of fit over scale",
            "dependsOnEarlierCopy": False,
            "singleSloganContractSatisfied": True,
        },
        **logo_policy_creator_extras(),
    }


def metaphorical_embodiment_judge_extras() -> Dict[str, Any]:
    return {
        "metaphoricalEmbodimentAssessment": {
            "literalExecutionDetected": False,
            "literalPresentationMeaningfullyTransformed": True,
            "creativeEmbodimentModeAccepted": True,
            "physicalEmbodimentMatchesStrategicRelationship": True,
            "viewerDiscoveryPresent": True,
            "sloganOnlyBridgesNotExplains": True,
            "creativeEmbodimentAccepted": True,
            "rejectionReason": None,
        },
        "visualBridgeAssessment": {
            "centralVisibleDetail": "The visible closing of distance between two people",
            "sloganConnectionToVisibleDetail": "The visible gesture carries the meaning before copy appears",
            "sloganConnectionToRelativeAdvantage": "The slogan closes the strategic advantage without repeating the scene",
            "dependsOnEarlierCopy": False,
            "singleSloganContractSatisfied": True,
        },
        **logo_policy_judge_extras(),
    }


def metaphorical_embodiment_shortening_extras() -> Dict[str, Any]:
    return metaphorical_embodiment_creator_extras(
        strategic_perception="The product makes a long process shorter without losing substance.",
    ) | {
        "metaphoricalEmbodiment": {
            **metaphorical_embodiment_creator_extras()["metaphoricalEmbodiment"],
            "strategicPerception": "The product makes a long process shorter without losing substance.",
            "obviousLiteralVisualSymbols": ["timeline chart", "progress bar"],
            "literalSymbolDisposition": "rejected",
            "literalSymbolsRejectedOrTransformed": "Reject charts; shorten a visibly long physical line or queue.",
            "creativeEmbodimentMode": "external_metaphor",
            "embodimentSubjectOrWorld": "A studio table with physical measuring tools",
            "physicalEmbodiment": "A long ribbon is cut to a shorter finished length",
            "embodiedStrategicRelationship": "Visible shortening expresses strategic time compression",
            "visiblePhysicalRelationship": "The cut length difference is readable without copy",
            "transformationMechanism": "Material length is shortened instead of showing a timeline widget",
            "whyTheVisualIsNotLiteralExplanation": "Time is shown through material length, not interface widgets",
            "understandableBeforeSlogan": True,
            "sloganBridgeToBusinessMeaning": "The slogan connects the shortened ribbon to faster delivery",
        },
        "visualBridgeAssessment": {
            "centralVisibleDetail": "The shortened ribbon beside the uncut length",
            "sloganConnectionToVisibleDetail": "The slogan names the shortening already visible",
            "sloganConnectionToRelativeAdvantage": "The slogan closes the speed advantage",
            "dependsOnEarlierCopy": False,
            "singleSloganContractSatisfied": True,
        },
        **logo_policy_creator_extras(),
    }


def metaphorical_embodiment_absence_extras() -> Dict[str, Any]:
    return {
        "metaphoricalEmbodiment": {
            "strategicPerception": "What is missing reveals the forgotten step.",
            "obviousLiteralVisualSymbols": ["checklist UI", "task reminder notification"],
            "literalSymbolDisposition": "rejected",
            "literalSymbolsRejectedOrTransformed": "Reject UI; show absence through an incomplete physical routine.",
            "creativeEmbodimentMode": "absence_or_planned_absurdity",
            "embodimentSubjectOrWorld": "A dark room with visible activity continuing",
            "physicalEmbodiment": "Someone acts normally while the room stays dark",
            "embodiedStrategicRelationship": "The missing light reveals the forgotten action",
            "visiblePhysicalRelationship": "Darkness persists despite movement",
            "transformationMechanism": "The omitted switch leaves darkness as the visible proof",
            "whyTheVisualIsNotLiteralExplanation": "Absence is shown through the scene, not a reminder app",
            "understandableBeforeSlogan": True,
            "sloganBridgeToBusinessMeaning": "The slogan connects the visible absence to the business insight",
        },
        "visualBridgeAssessment": {
            "centralVisibleDetail": "Activity continuing in darkness",
            "sloganConnectionToVisibleDetail": "The slogan completes the forgotten-step inference",
            "sloganConnectionToRelativeAdvantage": "The slogan closes why noticing absence matters",
            "dependsOnEarlierCopy": False,
            "singleSloganContractSatisfied": True,
        },
        **logo_policy_creator_extras(),
    }


def creative_embodiment_think_small_extras() -> Dict[str, Any]:
    return {
        "metaphoricalEmbodiment": {
            "strategicPerception": "Smallness becomes the visible advantage when scale and space are transformed.",
            "obviousLiteralVisualSymbols": ["vehicle spec sheet", "size comparison chart"],
            "literalSymbolDisposition": "transformed",
            "literalSymbolsRejectedOrTransformed": "Reject charts; show the actual product tiny in a vast white field.",
            "creativeEmbodimentMode": "transformed_product",
            "embodimentSubjectOrWorld": "The advertised compact car itself in a vast white field",
            "physicalEmbodiment": "The real product sits very small while empty space dominates the frame",
            "embodiedStrategicRelationship": "Visible scale inversion turns smallness into perceptual advantage",
            "visiblePhysicalRelationship": "The tiny product against expansive white space is readable before copy",
            "transformationMechanism": "Composition and scale transform the real weakness into the visible idea",
            "whyTheVisualIsNotLiteralExplanation": "The frame does the inversion; no measurement label explains it",
            "understandableBeforeSlogan": True,
            "sloganBridgeToBusinessMeaning": "Think Small completes the scale inversion the viewer already felt",
        },
        "visualBridgeAssessment": {
            "centralVisibleDetail": "The tiny car isolated in vast white space",
            "sloganConnectionToVisibleDetail": "The slogan names the scale inversion already visible",
            "sloganConnectionToRelativeAdvantage": "The slogan closes why smallness is the advantage",
            "dependsOnEarlierCopy": False,
            "singleSloganContractSatisfied": True,
        },
        **logo_policy_creator_extras(),
    }


def creative_embodiment_media_replacement_extras() -> Dict[str, Any]:
    return {
        "metaphoricalEmbodiment": {
            "strategicPerception": "The advertising medium itself can prove the strategic claim.",
            "obviousLiteralVisualSymbols": ["generic billboard layout", "standard print ad frame"],
            "literalSymbolDisposition": "transformed",
            "literalSymbolsRejectedOrTransformed": "Reject generic layout; make the medium itself the proof object.",
            "creativeEmbodimentMode": "transformed_medium",
            "embodimentSubjectOrWorld": "The storefront window display surface",
            "physicalEmbodiment": "The display format becomes the persuasive evidence",
            "embodiedStrategicRelationship": "The medium transformation proves closeness to the customer",
            "visiblePhysicalRelationship": "The format change is visible before any slogan appears",
            "transformationMechanism": "The container stops being neutral and becomes the argument",
            "whyTheVisualIsNotLiteralExplanation": "The medium works as proof, not as decoration",
            "understandableBeforeSlogan": True,
            "sloganBridgeToBusinessMeaning": "The slogan completes what the transformed medium already showed",
        },
        "visualBridgeAssessment": {
            "centralVisibleDetail": "The display surface acting as proof",
            "sloganConnectionToVisibleDetail": "The slogan bridges the visible medium transformation",
            "sloganConnectionToRelativeAdvantage": "The slogan closes the strategic advantage",
            "dependsOnEarlierCopy": False,
            "singleSloganContractSatisfied": True,
        },
        **logo_policy_creator_extras(),
    }


def creative_embodiment_context_collision_extras() -> Dict[str, Any]:
    return {
        "metaphoricalEmbodiment": {
            "strategicPerception": "Extreme context contrast makes the product role visible.",
            "obviousLiteralVisualSymbols": ["category comparison chart", "feature checklist"],
            "literalSymbolDisposition": "transformed",
            "literalSymbolsRejectedOrTransformed": "Reject charts; place the product in a sharply transformed environment.",
            "creativeEmbodimentMode": "transformed_context",
            "embodimentSubjectOrWorld": "The product remains visible inside a hostile or extreme setting",
            "physicalEmbodiment": "The familiar product persists while the surrounding context collides with expectation",
            "embodiedStrategicRelationship": "Context collision reveals the product's distinctive role",
            "visiblePhysicalRelationship": "The contrast between product and environment is readable silently",
            "transformationMechanism": "Environment transformation reassigns meaning to the visible product",
            "whyTheVisualIsNotLiteralExplanation": "The collision creates meaning through placement, not annotation",
            "understandableBeforeSlogan": True,
            "sloganBridgeToBusinessMeaning": "The slogan completes the collision the viewer already noticed",
        },
        "visualBridgeAssessment": {
            "centralVisibleDetail": "The product visible inside the colliding context",
            "sloganConnectionToVisibleDetail": "The slogan bridges the visible contrast",
            "sloganConnectionToRelativeAdvantage": "The slogan closes the strategic promise",
            "dependsOnEarlierCopy": False,
            "singleSloganContractSatisfied": True,
        },
        **logo_policy_creator_extras(),
    }


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
        **complete_ad_creator_extras(
            product_name=str((strategy or {}).get("productNameResolved") or "ACE Product"),
            language=str((strategy or {}).get("language") or "en"),
        ),
        **metaphorical_embodiment_creator_extras(),
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
        **complete_ad_creator_extras(
            product_name="ACE Product",
            language=str((strategy or {}).get("language") or "en"),
        ),
        **metaphorical_embodiment_creator_extras(),
        **_prototype_application(prototype_id),
    }


def complete_ad_judgment_extras(*, prototype_id: str = "closest") -> Dict[str, Any]:
    return {
        "semanticAlignmentAssessment": {
            "visualMeaning": "Physical closeness is visible in the film plan",
            "sloganMeaning": "The slogan closes the same strategic closeness advantage",
            "combinedAdvertisingMeaning": "Together they communicate closeness as the product promise",
            "sameStrategicPromise": True,
            "sloganCompletesRatherThanChangesVisual": True,
            "understandableWithoutCreatorReport": True,
            "keyWordMeaningsConnected": True,
            "semanticAlignment": True,
            "failureReason": None,
        },
        "prototypeApplicationAssessment": {
            "assignedPrototypeId": prototype_id,
            "prototypeMethodVisibleInFilm": True,
            "prototypeMethodReinforcedBySlogan": True,
            "applicationFeelsIntrinsic": True,
            "applicationRequiresRetrospectiveExplanation": False,
            "prototypeFitScore": 12,
        },
    }


def methodology_judgment_extras(*, prototype_id: str = "closest") -> Dict[str, Any]:
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
        "advertisingCompletionAssessment": {
            "advertiserIdentifiable": True,
            "productNamePresent": True,
            "relativeAdvantageClosed": True,
            "sloganSpecificToIdea": True,
            "functionsAsAdvertisement": True,
            "notes": "The Creator slogan closes the same relative advantage embodied by the film.",
        },
        **complete_ad_judgment_extras(prototype_id=prototype_id),
        **advertising_slogan_quality_judge_extras(),
        **metaphorical_embodiment_judge_extras(),
    }


def methodology_winner_extras(
    *,
    headline_decision: str = "include",
    winning_candidate: Dict[str, Any] | None = None,
    strategy: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    strategy_obj = strategy or methodology_strategy_extras()
    candidate = winning_candidate or methodology_candidate_extras("closest", strategy=strategy_obj)
    relative_advantage_source = str((strategy_obj.get("relativeAdvantage") or {}).get("statement") or "").strip()
    closure = (candidate.get("advertisingClosure") or {}) if isinstance(candidate.get("advertisingClosure"), dict) else {}
    final_slogan = str(closure.get("sloganText") or "קרוב יותר ממה שחשבת").strip()
    preservation = {
        "strategyFoundationId": strategy_obj.get("strategyFoundationId"),
        "prototypeId": candidate.get("prototypeId"),
        "structureType": candidate.get("structureType"),
        "visualParallelType": candidate.get("visualParallelType"),
        "coreCreativeMechanism": candidate.get("coreCreativeMechanism"),
    }
    headline_form = "none" if headline_decision == "omit" else "direct"
    canonical_decision = "omit" if headline_decision == "omit" else "use"
    return {
        "methodologyVersion": METHODOLOGY_VERSION,
        "headlineDecision": {
            "decision": canonical_decision,
            "reason": None,
            "reasonSource": "not_required",
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
        **advertising_slogan_quality_winner_extras(
            relative_advantage_source=relative_advantage_source,
            final_slogan_text=final_slogan,
        ),
    }


def single_slogan_contract_extras(*, slogan_text: str = "קרוב יותר ממה שחשבת") -> Dict[str, Any]:
    return {
        "copyContractVersion": "builder2_single_slogan_v1",
        "logoPolicyVersion": "builder2_no_logos_v1",
        "logosAllowed": False,
        "advertisedLogoAllowed": False,
        "thirdPartyLogosAllowed": False,
        "inventedLogoAllowed": False,
        "plainTextAdvertisedNameAllowed": True,
        "plainTextAdvertisedNameOnly": True,
        "inSceneBrandTextAllowed": False,
        "builder2NewFormatVersion": "builder2_complete_ad_v1",
        "advertisingSloganQualityContractVersion": BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION,
        "sloganDecision": "use",
        "sloganText": slogan_text,
        "sloganCoreKeyword": "closer",
        "sloganSource": "creator_candidate_closure",
        "sloganUnderstandsWithoutPriorCopy": True,
        "sloganRenderedExactlyOnce": False,
        "headlineOverlaySkipped": True,
        "headlineCompatibilityAlias": True,
    }

