"""
Builder2 tournament prompt builders — isolated from legacy video_planning prompts.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from engine.builder2_creator_core_contract import build_creator_required_keys_prompt_text
from engine.builder2_methodology_contract import (
    CREATIVE_STAGE_ORDER,
    INTEREST_PRIORITY_ORDER,
    METHODOLOGY_VERSION,
    TOURNAMENT_MOTTO,
    VALID_CONTINUITY_RISK,
    VALID_HEADLINE_DECISIONS,
    VALID_HEADLINE_FORMS,
    VALID_STRUCTURE_TYPES,
    VALID_VISUAL_PARALLEL_TYPES,
    prompt_enum_list,
)
from engine.builder2_judge_core_contract import (
    build_judge_factual_grounding_prompt_text,
    build_judge_required_keys_prompt_text,
    resolve_creator_verbal_decision,
)
from engine.builder2_prototypes import Builder2Prototype
from engine.builder2_runway_config import resolve_builder2_video_duration_seconds
from engine.builder2_strategy_identity import expected_strategy_foundation_id
from engine.builder2_tournament_contracts import (
    CANDIDATE_SCHEMA_VERSION,
    JUDGMENT_SCHEMA_VERSION,
    JUDGE_SCORE_RANGES,
    STRATEGY_SCHEMA_VERSION,
    VALID_GROUNDING_TYPES,
    WINNER_PLAN_SCHEMA_VERSION,
)


def build_strategy_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
) -> str:
    grounding_types = ", ".join(sorted(VALID_GROUNDING_TYPES))
    stage_order = " → ".join(CREATIVE_STAGE_ORDER)
    interest_order = " → ".join(INTEREST_PRIORITY_ORDER)
    return (
        "You are the Builder2 Strategy role generating ONE fixed strategic foundation.\n"
        "You are a methodologist, not a copywriter or director.\n"
        "Do NOT choose a prototype. Do NOT create a visual concept, headline, keyword, object pair, or Runway prompt.\n"
        "Do NOT invent statistics, studies, or customer research.\n"
        "Ground the problem in real observable practice, physical reality, market behavior, or professional knowledge.\n"
        "Perceptual buyer problems are valid when grounded in observable practice or market behavior.\n"
        "Identify ONE grounded problem only. Reject generic goals such as wanting more customers or needing awareness.\n"
        f"Creative order for downstream roles: {stage_order}.\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Language: {language}\n\n"
        f"Return one JSON object only with schemaVersion={STRATEGY_SCHEMA_VERSION!r}, "
        f"methodologyVersion={METHODOLOGY_VERSION!r}, and keys:\n"
        "productNameResolved, language, problemPerception{statement,groundingType,groundingEvidence,whyItMatters}, "
        "relativeAdvantage{statement,derivationFromProblem,truthBoundary,admitsRelevantGap,"
        "relativeAdvantageType,relativeAdvantageEvidence,relativeAdvantageEvidenceSourcePaths,"
        "relativeAdvantageInferenceLevel,categoryConventionDependencies,unsupportedAssumptions,"
        "relativeAdvantageFactuallyGrounded}, "
        "strategyEvidenceGrounding{contractVersion,productMarketStatus,productInformationDensity,"
        "explicitProductFacts,safeStrategicInterpretations,categoryConventions,unsupportedAssumptions}, "
        "mechanismScan{domainFacts,discoveredMechanism,creativeOpportunity,depthEvidence}.\n"
        f'language must be exactly "he" or "en".\n'
        f"groundingType must be exactly one of: {grounding_types}.\n"
        "groundingEvidence must be a non-empty JSON array of concise qualitative evidence strings; "
        "one strong item is sufficient. Do not invent statistics.\n"
        "domainFacts must be a non-empty JSON array of concise qualitative domain facts; "
        "professional knowledge and common market behavior are acceptable without citations.\n"
        "truthBoundary must explain what the advantage does not falsely claim.\n"
        "admitsRelevantGap must be JSON boolean true or false.\n"
        "depthEvidence must explain why the discovered mechanism is deeper than a first association.\n"
        "Evidence grounding contract: distinguish explicit product facts, safe strategic interpretation, "
        "category convention, and unsupported product claims.\n"
        "Do NOT convert agency revision rounds, campaign optimization, feedback loops, learning from results, "
        "or ongoing account management into product capabilities unless explicitly supplied in the product description.\n"
        "When product information is sparse, prelaunch, or unknown, keep relativeAdvantageInferenceLevel at "
        "explicit or direct_derivation only and set relativeAdvantageFactuallyGrounded=true.\n"
        "Category conventions may inform mechanismScan but must not become product promises.\n"
        "Limited information does not justify inventing operational features; use verified facts plus creative metaphor.\n"
        "Reference-only methods such as market-code inversion may inform mechanism scan guidance but must not become visual concepts here.\n"
        'If you cannot ground a valid problem, return {"planningFailure":"builder2_strategy_not_grounded"} only.'
    )


def build_strategy_repair_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    invalid_output: Dict[str, Any],
    validation_failures: List[str],
) -> str:
    return (
        "You are the Builder2 Strategy repair role.\n"
        "Repair ONLY the listed validation defects in the strategic foundation JSON.\n"
        "Do NOT choose a prototype. Do NOT create a visual concept, headline, or Runway prompt.\n"
        "Do NOT invent statistics, studies, or customer research.\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Language: {language}\n\n"
        "Original strategy instructions:\n"
        f"{build_strategy_prompt(product_name=product_name, product_description=product_description, language=language)}\n\n"
        "Invalid structured output to repair:\n"
        f"{json.dumps(invalid_output, ensure_ascii=False)}\n\n"
        "Exact validation failures to fix:\n"
        + "\n".join(f"- {item}" for item in validation_failures)
        + "\n\n"
        f"Return one repaired JSON object only with schemaVersion={STRATEGY_SCHEMA_VERSION!r}."
    )


def build_creator_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype: Builder2Prototype,
    candidate_id: str,
    attempt_number: int,
    runway_mode: str,
) -> str:
    duration = resolve_builder2_video_duration_seconds()
    structure_types = prompt_enum_list(VALID_STRUCTURE_TYPES)
    continuity_risks = prompt_enum_list(VALID_CONTINUITY_RISK)
    visual_parallel_types = prompt_enum_list(VALID_VISUAL_PARALLEL_TYPES)
    stage_order = " → ".join(CREATIVE_STAGE_ORDER)
    interest_order = " → ".join(INTEREST_PRIORITY_ORDER)
    strategy_id = expected_strategy_foundation_id(strategy_foundation)
    strategy_digest = strategy_foundation.get("strategyFoundationDigest") or ""
    return (
        "You are the Builder2 Creator role generating ONE isolated candidate idea.\n"
        f"Motto: {TOURNAMENT_MOTTO}\n"
        "Interpret the prototype by how it addressed a problem, not by the original problem, object or surface appearance.\n"
        "Do NOT start from a headline, clever word, random object, or visual trick and invent strategy afterward.\n"
        f"Follow this workflow while developing the candidate: {stage_order}.\n"
        "Do not output private reasoning or a narrative of your thought process.\n"
        "Return only the required structured conclusions.\n"
        "The server enforces the workflow contract; do not self-certify internal reasoning order.\n"
        f"Interest priority: {interest_order}.\n"
        "Do not optimize for Runway simplicity before finding an interesting mechanism.\n"
        "Do not use arbitrary weirdness as interest. Freshness must come from a deep mechanism.\n"
        "Realism and silent clarity are mandatory constraints. Everyday is lowest priority, not the starting point.\n"
        "Creators inherit the Strategy evidence ledger. You may vary prototype method and visual expression freely, "
        "but you must not introduce new product capabilities, strengthen qualified claims into absolute claims, "
        "or convert category knowledge into a product promise.\n"
        "Sparse product information still permits rich visual metaphor; it does not permit inventing feedback, "
        "revision, optimization, learning, or improvement workflows unless explicitly supplied.\n"
        "You know ONLY this assigned prototype and this attempt ID.\n"
        "Do NOT reference previous candidates, Judge scores, tournament standings, or other prototypes.\n"
        "Do NOT output a final headline, Runway production prompt, marketing copy, image request, or Judge score.\n"
        f"Candidate ID: {candidate_id}\n"
        f"Attempt number: {attempt_number}\n"
        f"Assigned prototype ID: {prototype.prototype_id}\n"
        f"Prototype display name: {prototype.display_name}\n"
        f"Original problem: {prototype.original_problem}\n"
        f"Reusable method: {prototype.reusable_method}\n"
        f"Do not copy: {prototype.must_not_copy}\n"
        f"Creator guidance: {prototype.creator_guidance}\n"
        "The assigned prototype method is supplied by the server. Do not restate or redefine the prototype method.\n"
        "Apply the method to the current Strategy Foundation through the required prototype-specific application object, "
        "visualMechanism, creatorReport.whyParallelExpressesAdvantage, and silent video execution.\n"
        "Return structured application evidence, not a summary of the prototype instructions.\n"
        f"Video duration seconds: {duration}\n"
        f"Runway mode constraint: {runway_mode}\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Language: {language}\n"
        f"strategyFoundationId (return exactly): {strategy_id!r}\n"
        f"strategyFoundationDigest (reference only, do not recalculate): {strategy_digest!r}\n"
        "Fixed strategic foundation (unchanged for all candidates):\n"
        f"{json.dumps(strategy_foundation, ensure_ascii=False)}\n\n"
        f"Return one JSON object only with schemaVersion={CANDIDATE_SCHEMA_VERSION!r}, "
        f"methodologyVersion={METHODOLOGY_VERSION!r}. No Markdown fences. No prose.\n"
        f"{build_creator_required_keys_prompt_text(prototype_id=prototype.prototype_id)}\n"
        f"strategyFoundationId must be exactly {strategy_id!r}.\n"
        f"prototypeId must be exactly {prototype.prototype_id!r}.\n"
        f"structureType must be exactly one of: {structure_types}.\n"
        f"continuityRisk must be exactly one of: {continuity_risks}.\n"
        f"visualParallelType must be exactly one of: {visual_parallel_types}. "
        'If using "other", explain the parallel clearly in creatorReport.whyParallelExpressesAdvantage.\n'
        "verbalPotential.decision must be exactly one of: available, not_needed, not_found.\n"
        "When decision=available provide keywordOrKeyPhrase, visualMeaning, strategicMeaning, and bornFromVisibleMechanism=true.\n"
        "When decision=not_needed or not_found provide a short reason.\n"
        "creatorReport.silentVerification may be a string explanation OR use silentVerification{understandableWithoutAudio,explanation}.\n"
        "runwayFeasibility.generationRisks must be a JSON array (empty array allowed when risk is low).\n"
        f"creatorReport.goldPrototypeUsed must be {prototype.prototype_id!r} or {prototype.display_name!r}.\n"
        "Hebrew free-text fields are allowed. Enum fields must use the exact canonical English tokens above.\n"
        "For think_small identify a real weakness. For essential_pairing avoid appearance-only pairing.\n"
        "For context_collision include a meaningful bridge explanation in creatorReport.\n"
        "Metaphorical embodiment is mandatory: do not illustrate the strategic perception with literal domain symbols "
        "(graphs, dashboards, printed reports, CRM screens, forms, counters, growth arrows) unless they undergo a "
        "strong conceptual transformation. Create a physical visual embodiment through which the viewer experiences "
        "or discovers the strategic perception silently before advertisingClosure.sloganText appears. "
        "The product, medium, or business context may remain visible when scale, composition, context, motion, "
        "absence, collision, or medium transformation does the conceptual work — as in Think Small. "
        "Do not default to quantity-versus-quality, external worlds, or repeated-object comparisons unless that "
        "relationship is genuinely the candidate's strategic perception. "
        "No-logo policy: the Runway visual must be entirely unbranded; generic objects only; no logos, wordmarks, "
        "emblems, monograms, badges, watermarks, branded packaging/clothing/screens/vehicles/signs, invented marks, "
        "or visible commercial names in-frame. Reserve the advertised name for plain text on the closure card only. "
        "dependsOnEarlierCopy must be false in visualBridgeAssessment."
    )


def build_creator_repair_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype: Builder2Prototype,
    candidate_id: str,
    attempt_number: int,
    runway_mode: str,
    invalid_output: Dict[str, Any],
    validation_failures: List[str],
) -> str:
    slogan_repair = any("sloganText.word_limit" in item for item in validation_failures)
    slogan_repair_block = ""
    if slogan_repair:
        from engine.builder2_advertising_closure_contract import SLOGAN_MAX_WORD_COUNT, build_slogan_word_limit_prompt_text

        slogan_repair_block = (
            "\nSlogan-only repair scope:\n"
            f"- Shorten advertisingClosure.sloganText to at most {SLOGAN_MAX_WORD_COUNT} words using the server counting rule.\n"
            "- Do NOT truncate mechanically by deleting words at the end; rewrite a shorter slogan that preserves Hebrew meaning, "
            "syntax, and wordplay when applicable.\n"
            "- Preserve problemPerception, relativeAdvantage, prototype method application, visualMechanism, scene, visual anchor, "
            "creative embodiment, no-logo plan, Runway feasibility, and all non-copy fields unchanged.\n"
            "- You may update only slogan copy and directly dependent slogan metadata such as semanticBridge slogan fields, "
            "metaphoricalEmbodiment.sloganBridgeToBusinessMeaning, visualBridgeAssessment slogan connections, "
            "verbalPotential keyword/meanings when they must stay aligned, and compatible headline alias fields that must equal "
            "the canonical slogan.\n"
            "- Keep exactly one canonical slogan; dependsOnEarlierCopy must remain false; do not add a headline or second copy layer.\n"
            f"{build_slogan_word_limit_prompt_text()}\n"
            "- Return a JSON object with top-level key sloganRepairPatch containing ONLY fields that must change.\n"
            "- Required: advertisingClosure.sloganText.\n"
            "- Optional: semanticBridge.sloganMeaning, semanticBridge.howTheMeaningsMeet, "
            "metaphoricalEmbodiment.sloganBridgeToBusinessMeaning, "
            "visualBridgeAssessment.sloganConnectionToVisibleDetail, visualBridgeAssessment.sloganConnectionToRelativeAdvantage, "
            "verbalPotential.keywordOrKeyPhrase, verbalPotential.strategicMeaning.\n"
            "- Do NOT repeat unchanged fields; omit any field that can remain from the original candidate.\n"
            "- Do NOT return a full Creator candidate; the server merges sloganRepairPatch into the original candidate.\n"
        )
    return (
        "You are the Builder2 Creator repair role.\n"
        "Repair ONLY the listed structural/schema defects. Preserve the creative idea.\n"
        "Do NOT reference other candidates, Judge scores, or tournament standings.\n"
        f"Candidate ID: {candidate_id}\n"
        f"Assigned prototype ID: {prototype.prototype_id}\n\n"
        "Original Creator instructions:\n"
        f"{build_creator_prompt(product_name=product_name, product_description=product_description, language=language, strategy_foundation=strategy_foundation, prototype=prototype, candidate_id=candidate_id, attempt_number=attempt_number, runway_mode=runway_mode)}\n\n"
        "Invalid structured output to repair:\n"
        f"{json.dumps(invalid_output, ensure_ascii=False)}\n\n"
        "Exact validation failures to fix:\n"
        + "\n".join(f"- {item}" for item in validation_failures)
        + slogan_repair_block
        + "\n\n"
        + (
            f"Return one JSON object only with schemaVersion={CANDIDATE_SCHEMA_VERSION!r} and top-level sloganRepairPatch."
            if slogan_repair
            else f"Return one repaired JSON object only with schemaVersion={CANDIDATE_SCHEMA_VERSION!r}."
        )
    )


def build_semantic_bridge_repair_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype: Builder2Prototype,
    candidate_id: str,
    base_candidate: Dict[str, Any],
    validation_failures: List[str],
) -> str:
    allowed_paths = ", ".join(
        (
            "semanticBridge.keyWordOrConcept",
            "semanticBridge.visualMeaning",
            "semanticBridge.strategicMeaning",
            "semanticBridge.sloganMeaning",
            "semanticBridge.howTheMeaningsMeet",
            "semanticBridge.understandableWithoutCreatorReport",
            "semanticBridge.dualMeaningUsed",
            "semanticBridge.physicalMeaningActivatedByVisual",
            "semanticBridge.strategicMeaningActivatedBySlogan",
            "semanticBridge.meaningsConverge",
            "visualBridgeAssessment.sloganConnectionToVisibleDetail",
            "visualBridgeAssessment.sloganConnectionToRelativeAdvantage",
            "metaphoricalEmbodiment.sloganBridgeToBusinessMeaning",
            "verbalPotential.keywordOrKeyPhrase",
            "verbalPotential.strategicMeaning",
        )
    )
    return (
        "You are the Builder2 Creator semantic-bridge repair role.\n"
        "Complete the semantic bridge between the preserved visual mechanism, preserved strategic meaning, "
        "and the already repaired seven-word slogan.\n"
        "Do NOT generate a new Creator candidate.\n"
        "Do NOT change the slogan, visual execution, strategic problem, relative advantage, prototype method, "
        "scene, Runway feasibility, no-logo plan, product name, or any non-bridge field.\n"
        f"Candidate ID: {candidate_id}\n"
        f"Assigned prototype ID: {prototype.prototype_id}\n\n"
        "Immutable preserved candidate context:\n"
        f"{json.dumps(base_candidate, ensure_ascii=False)}\n\n"
        "Exact validation failures to fix:\n"
        + "\n".join(f"- {item}" for item in validation_failures)
        + "\n\nSemantic-bridge repair scope:\n"
        "- Return a JSON object with top-level key semanticBridgeRepairPatch containing ONLY fields that must change.\n"
        "- Provide substantive semantic/verbal bridge fields that establish why the preserved visual meaning and "
        "preserved slogan meaning converge on the preserved relative advantage.\n"
        "- Do NOT merely set meaningsConverge=true without completing the required semantic fields.\n"
        "- Do NOT change advertisingClosure.sloganText.\n"
        f"- Permitted paths only: {allowed_paths}.\n"
        "- Omit unchanged fields.\n"
        f"Return one JSON object only with top-level semanticBridgeRepairPatch."
    )


def build_creator_retry_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype: Builder2Prototype,
    candidate_id: str,
    attempt_number: int,
    runway_mode: str,
    retry_rule: str,
) -> str:
    return (
        "You are the Builder2 Creator role generating ONE fresh isolated candidate idea.\n"
        "This is a clean retry for the same assigned prototype slot.\n"
        "Do NOT reference any previous candidate output, Judge scores, or tournament standings.\n"
        f"Candidate ID: {candidate_id}\n"
        f"Assigned prototype ID: {prototype.prototype_id}\n"
        f"Methodology rule to satisfy: {retry_rule}\n\n"
        f"{build_creator_prompt(product_name=product_name, product_description=product_description, language=language, strategy_foundation=strategy_foundation, prototype=prototype, candidate_id=candidate_id, attempt_number=attempt_number, runway_mode=runway_mode)}"
    )


def build_judge_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype: Builder2Prototype,
    candidate: Dict[str, Any],
    candidate_id: str,
) -> str:
    interest_order = " → ".join(INTEREST_PRIORITY_ORDER)
    contract = candidate.get("prototypeMethodContract") or {}
    creator_verbal_decision = resolve_creator_verbal_decision(candidate)
    judge_contract = build_judge_required_keys_prompt_text(
        creator_verbal_decision=creator_verbal_decision,
        candidate_id=candidate_id,
    )
    return (
        "You are the Builder2 Judge role evaluating ONE candidate independently.\n"
        f"Motto: {TOURNAMENT_MOTTO}\n"
        "Do NOT reward prototype prestige, literal similarity, historical fame, or prototype ID.\n"
        "Do NOT reward repetition of the prototype description or any Creator-written methodSummary.\n"
        "Judge whether the actual visual mechanism applied the assigned prototype's problem-solving method "
        "to the current Problem Perception and Relative Advantage. Judge the application itself.\n"
        "Do NOT redesign the idea, generate a replacement advertisement, or compare to unseen candidates.\n"
        "Do NOT infer missing Creator intent beyond the candidate and Creator Report.\n"
        "Assume the idea is wrong until it proves itself. A valid eligible=false judgment is valid.\n"
        f"Interest priority for qualitative assessment: {interest_order}.\n"
        f"Candidate ID: {candidate_id}\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Language: {language}\n"
        "Fixed strategic foundation:\n"
        f"{json.dumps(strategy_foundation, ensure_ascii=False)}\n"
        "Canonical prototype method contract (authoritative):\n"
        f"{json.dumps(contract, ensure_ascii=False)}\n"
        "Assigned prototype definition (instruction context only — not proof of application):\n"
        f"{json.dumps({'prototypeId': prototype.prototype_id, 'displayName': prototype.display_name, 'originalProblem': prototype.original_problem, 'reusableMethod': prototype.reusable_method, 'mustNotCopy': prototype.must_not_copy, 'judgeQualityGuidance': prototype.judge_quality_guidance}, ensure_ascii=False)}\n"
        "Candidate to judge:\n"
        f"{json.dumps(candidate, ensure_ascii=False)}\n\n"
        f"Return one JSON object only with schemaVersion={JUDGMENT_SCHEMA_VERSION!r}, "
        f"methodologyVersion={METHODOLOGY_VERSION!r}. No Markdown fences. No prose.\n"
        f"candidateId must be exactly {candidate_id!r}.\n"
        "eligible must be JSON boolean true or false, not a string.\n"
        "disqualifiers must be a JSON array.\n"
        "strengths must be a JSON array.\n"
        "weaknesses must be a JSON array.\n"
        "confidence must be a JSON number from 0.0 to 1.0.\n"
        "Every score must be an integer within its category maximum.\n"
        "Do NOT output totalScore, total, or any authoritative total score field.\n"
        "Hebrew free-text fields are allowed in verdict, strengths, weaknesses and prototypeQualityComparison.\n"
        f"{judge_contract}\n"
        "Required keys: candidateId, eligible, disqualifiers, scores, verdict, strengths, weaknesses, "
        "prototypeQualityComparison, confidence, problemAdvantageAssessment, mechanismDepthAssessment, "
        "prototypeMethodAssessment, visualMechanismAssessment, participationAssessment, visualFamilyAssessment, "
        "silentMovieAssessment, verbalLayerAssessment, headlineNecessityAssessment, advertisingCompletionAssessment, "
        "factualGroundingAssessment.\n"
        f"{build_judge_factual_grounding_prompt_text()}\n"
        "Independently test creative embodiment: reject literal graph/report/dashboard/interface execution unless "
        "meaningfully transformed; require metaphoricalEmbodimentAssessment and visualBridgeAssessment with "
        "dependsOnEarlierCopy=false. Do not require leaving the business domain. "
        "Clarity without creative embodiment is insufficient. "
        "Independently enforce the no-logo policy: reject logo-dependent concepts, real or invented logos, "
        "third-party branding, in-scene brand text, and stylized wordmarks.\n"
        "If eligible=false, include at least one disqualifier explaining why."
    )


def build_judge_repair_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype: Builder2Prototype,
    candidate: Dict[str, Any],
    candidate_id: str,
    invalid_output: Dict[str, Any],
    validation_failures: List[str],
) -> str:
    return (
        "You are the Builder2 Judge repair role.\n"
        "Repair ONLY the listed structural defects. Preserve the substantive judgment.\n"
        "Do NOT redesign the candidate or change eligibility merely to satisfy schema.\n"
        "When factualGroundingAssessment is structurally defective, complete ONLY that object while preserving "
        "scores, verdict, strengths, weaknesses, confidence, and all other already-valid assessments.\n"
        f"{build_judge_factual_grounding_prompt_text()}\n"
        f"Candidate ID: {candidate_id}\n\n"
        "Original Judge instructions:\n"
        f"{build_judge_prompt(product_name=product_name, product_description=product_description, language=language, strategy_foundation=strategy_foundation, prototype=prototype, candidate=candidate, candidate_id=candidate_id)}\n\n"
        "Invalid structured output to repair:\n"
        f"{json.dumps(invalid_output, ensure_ascii=False)}\n\n"
        "Exact validation failures to fix:\n"
        + "\n".join(f"- {item}" for item in validation_failures)
        + "\n\n"
        f"Return one repaired JSON object only with schemaVersion={JUDGMENT_SCHEMA_VERSION!r}."
    )


def build_judge_retry_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype: Builder2Prototype,
    candidate: Dict[str, Any],
    candidate_id: str,
    retry_rule: str,
) -> str:
    return (
        "You are the Builder2 Judge role performing ONE clean retry for the same candidate.\n"
        "Do NOT reference any previous Judge response, score, ranking, or unseen candidate.\n"
        f"Candidate ID: {candidate_id}\n"
        f"Violated Judge rule to respect: {retry_rule}\n\n"
        f"{build_judge_prompt(product_name=product_name, product_description=product_description, language=language, strategy_foundation=strategy_foundation, prototype=prototype, candidate=candidate, candidate_id=candidate_id)}"
    )


def build_winner_music_direction_prompt_text() -> str:
    """Builder2 Winner musicDirection instructions — short-ad soundtrack methodology."""
    return (
        "musicDirection object: prompt (non-empty creative direction for instrumental soundtrack), "
        "instrumentalOnly=true, immediateStart=true.\n"
        "This is music for a very short advertisement — not a conventional long-form composition.\n"
        "FULL ARRANGEMENT FROM FIRST BEAT: immediateStart means the music begins at once AND the "
        "essential full arrangement must already be present from the first beat. "
        "These are two separate requirements.\n"
        "Design the soundtrack for what must already be audible in the opening seconds, "
        "not for a long track that gradually develops. "
        "Within approximately the first 1–2 seconds, the listener should already hear the important "
        "musical foundations this ad needs.\n"
        "The advertisement is very short. Useful musical content must already be present in "
        "approximately the first 10–15 seconds, and especially in the first 1–2 seconds. "
        "Do not assume later portions of a generated track will ever be used in the final advertisement.\n"
        "Builder2 may use only the opening portion of a longer generated track; "
        "therefore the opening itself must already sound intentional, complete, layered, "
        "and professionally produced.\n"
        "Do not reserve essential rhythmic energy, low-frequency foundation, harmonic richness, "
        "melodic identity, percussion or pulse, or the main musical character for a later build-up "
        "unless a genuinely sparse/minimal creative direction is essential to the advertisement idea.\n"
        "Think in musical functions, not a fixed instrument list. When appropriate to the creative idea, "
        "consider whether the opening already includes foundations such as: low-frequency/bass support, "
        "rhythmic foundation, percussion or pulse, harmonic layer, melodic identity, and multiple "
        "complementary simultaneous layers. Do not force every function in every ad — "
        "avoid thin, undeveloped, or unintentionally single-layer soundtracks when the creative idea "
        "calls for richness.\n"
        "Gentle, quiet, elegant, restrained, intimate, subtle, soft, or delicate music does not have "
        "to be sparse or thin. A delicate soundtrack can still have complete bass foundation, "
        "rhythmic support, harmony, musical depth, and complementary layers. "
        "Do not treat soft/subdued/minimal wording as permission to strip the opening down to a lone "
        "instrument or thin texture unless sparse/minimal is truly the creative idea.\n"
        "Creative appropriateness remains mandatory: mood, energy, pacing, instrumentation, and musical "
        "character must fit this advertisement's specific mechanism, sequence, reveal, and closure — "
        "not a generic genre. Musical richness is not maximum loudness, maximum intensity, "
        "or the largest possible number of instruments.\n"
        "Optional short-ad timing guidance (conceptual only — no timestamps required): "
        "opening/action — full musical palette already active; "
        "middle — support development, change, tension, or reveal; "
        "closure — resolve or land naturally with the advertising ending.\n"
        "instrumental only; no vocals; no lyrics; no spoken words; no long intro; "
        "the soundtrack must start musically immediately.\n"
    )


def build_winner_development_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    winning_judgment: Dict[str, Any],
    prototype: Builder2Prototype,
    runway_mode: str,
    preservation_snapshot: Dict[str, Any],
) -> str:
    duration = resolve_builder2_video_duration_seconds()
    headline_forms = prompt_enum_list(VALID_HEADLINE_FORMS)
    headline_decisions = prompt_enum_list(VALID_HEADLINE_DECISIONS)
    from engine.builder2_advertising_slogan_quality_contract import (
        WINNER_SLOGAN_EVIDENCE_KEY,
        build_winner_advertising_slogan_prompt_text,
    )

    return (
        "You are the Builder2 Winner Developer converting ONE winning candidate into a production-ready video plan.\n"
        "Refine the winning execution only.\n"
        "Do not redesign the advertisement.\n"
        "Do not replace the visual mechanism.\n"
        "Do not solve weaknesses by creating a new concept.\n"
        "Preserve the winning creative mechanism exactly.\n"
        "Do NOT redesign the idea around motion, replace the visual family, replace the visual anchor, "
        "change the strategic problem, or change the relative advantage.\n"
        "Use editing and timing only to strengthen the same mechanism.\n"
        "First ask: How do I preserve the mechanism? Then: How do I express it through seven seconds of video?\n"
        "Generate the headline ONLY now when headlineDecision.decision=use (alias include). Headline remainder max seven words excluding product name.\n"
        "When headlineDecision.decision=omit, leave headline, headlineText, and headlineCoreKeyword empty and set headlineForm=none. "
        "Do not invent a separate in-scene headline; the authoritative end-card slogan is server-owned from the winning Creator candidate.\n"
        "Single-slogan contract (new jobs): set headlineDecision.decision=omit and headlineForm=none. "
        "Do not generate a separate in-video headline. advertisingClosure.sloganText from the winning candidate is the "
        "only advertising sentence and must bridge the finished visual mechanism.\n"
        "Required advertisingClosure object: required=true, productNameText, sloganText, language, "
        "presentationMode=end_card, durationSeconds=3.5, headlineSource, noLogo=true.\n"
        f"Video duration seconds: {duration}\n"
        f"Runway mode: {runway_mode}\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Language: {language}\n"
        "Fixed strategic foundation:\n"
        f"{json.dumps(strategy_foundation, ensure_ascii=False)}\n"
        "Winning candidate:\n"
        f"{json.dumps(winning_candidate, ensure_ascii=False)}\n"
        "Valid Judge judgment for this winning candidate only:\n"
        f"{json.dumps(winning_judgment, ensure_ascii=False)}\n"
        "Preservation snapshot (must match preservationReference identity fields exactly):\n"
        f"{json.dumps(preservation_snapshot, ensure_ascii=False)}\n"
        "Prototype method:\n"
        f"{json.dumps({'prototypeId': prototype.prototype_id, 'reusableMethod': prototype.reusable_method}, ensure_ascii=False)}\n\n"
        f"Return one JSON object only with schemaVersion={WINNER_PLAN_SCHEMA_VERSION!r}, "
        f"methodologyVersion={METHODOLOGY_VERSION!r}.\n"
        "Required keys: productNameResolved, language, prototypeId, "
        "coreCreativeMechanism, visualParallelType, visualFamily, structureType, headlineDecision, headlineForm, "
        f"advertisingClosure, {WINNER_SLOGAN_EVIDENCE_KEY}, coreVisualIdea, sequence{{beginning,development,resolution}}, sceneVariations, visualAnchor, "
        "openingFrameDescription, videoPrompt, musicDirection.\n"
        f"{build_winner_advertising_slogan_prompt_text()}\n"
        f"{build_winner_music_direction_prompt_text()}"
        "Optional diagnostic keys: preservationReference, winnerPreservationCheck, headlineDecision.reason.\n"
        "headlineDecision.decision is authoritative (use|omit; include accepted as alias for use). "
        "headlineDecision.reason is optional diagnostic metadata and is not required for validity.\n"
        "Strategic fields (problemPerception, relativeAdvantage, coreCreativeMechanism, prototype identity, "
        "visual anchor foundation) are server-owned and will be restored from the selected winning candidate.\n"
        f'headlineDecision.decision must be one of: {headline_decisions}.\n'
        f'headlineForm must be one of: {headline_forms}. headlineForm="none" requires decision="omit".\n'
        "If preservationReference is returned, it must match the source candidate identity fields.\n"
        'When decision="omit", headline may be empty.\n'
    )


def build_winner_headline_repair_prompt(
    *,
    product_name: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    winning_judgment: Dict[str, Any],
    parsed_winner_plan: Dict[str, Any],
    validation_failures: List[str],
) -> str:
    headline_form = str(parsed_winner_plan.get("headlineForm") or "").strip() or "(unspecified)"
    headline_decision = parsed_winner_plan.get("headlineDecision")
    scene_context = {
        "coreVisualIdea": parsed_winner_plan.get("coreVisualIdea"),
        "coreCreativeMechanism": parsed_winner_plan.get("coreCreativeMechanism"),
        "sequence": parsed_winner_plan.get("sequence"),
        "visualAnchor": parsed_winner_plan.get("visualAnchor"),
        "videoPrompt": parsed_winner_plan.get("videoPrompt"),
        "headlineForm": headline_form,
        "headlineDecision": headline_decision,
    }
    return (
        "You are the Builder2 Winner headline repair role.\n"
        "Repair ONLY the missing in-scene headline fields for an already accepted Winner plan.\n"
        "Preserve the existing advertisement exactly.\n"
        "Do NOT redesign, reinterpret, or replace the creative idea.\n"
        "Do NOT change the visual sequence, videoPrompt, strategy, mechanism, prototype, or advertising closure.\n"
        "Do NOT return a complete Winner plan.\n"
        "Do NOT copy the end-card slogan merely because it exists; the in-scene headline is a separate element.\n"
        "Do NOT depend on a fixed idiom unless the existing contract already permits it.\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Language: {language}\n"
        "Fixed strategic foundation:\n"
        f"{json.dumps(strategy_foundation, ensure_ascii=False)}\n"
        "Winning Creator candidate:\n"
        f"{json.dumps(winning_candidate, ensure_ascii=False)}\n"
        "Winning Judge judgment (headline necessity is authoritative):\n"
        f"{json.dumps(winning_judgment, ensure_ascii=False)}\n"
        "Existing parsed Winner plan scene context (preserve unchanged):\n"
        f"{json.dumps(scene_context, ensure_ascii=False)}\n"
        "Exact validation failures to fix:\n"
        + "\n".join(f"- {item}" for item in validation_failures)
        + "\n\n"
        "Return one JSON object only with exactly these keys:\n"
        '- "headline": non-empty remainder text excluding the product name; maximum seven words in the remainder.\n'
        '- "headlineCoreKeyword": exactly one meaningful word that appears in the headline remainder.\n'
        "The headline must be understandable in the requested language.\n"
        "No explanations outside JSON.\n"
    )
