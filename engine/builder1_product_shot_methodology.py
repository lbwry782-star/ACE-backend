"""
Builder1 creative methodology blocks — perception-first, anti product-shot bias.

Shared prompt text for planning stages. The model performs creative evaluation;
the server validates structure and model-marked eligibility only.
"""

BUILDER1_IDEA_BEFORE_PRODUCT = """
IDEA BEFORE PRODUCT — mandatory creative sequence:
1. Identify the strategic problem.
2. Select the relative advantage.
3. Create and fix the brand slogan.
4. Extract the action implied by the slogan.
5. Define the exact audience perception to create.
6. Search broadly for the clearest physical embodiment of that perception.
7. Select the conceptual generator.
8. Select a transferred physical generator.
9. Define the graphic generator.
10. Develop the ad series.

Do NOT begin from: the product shape, product package, product category, ordinary product use,
a conventional product photograph, or a desire to place the product at the center.
The physical embodiment must explain the perception best — not because it is the advertised object.
""".strip()

BUILDER1_PRODUCT_SHOT_BIAS = """
PRODUCT-SHOT BIAS:
Advertising models fall back to a product shot: large product, attractive lighting, clean background, campaign copy.
That is presentation — not necessarily an advertising idea.

Do NOT treat these as sufficient creative mechanisms:
- making the product larger, smaller, multiplied, recolored, stretched, or shortened;
- arranging many units of it;
- placing it on a dramatic background or surrounding it with decorative effects;
- showing attractive packaging;
- showing a person holding or using it;
- treating the product as the hero merely because it is being advertised.

A product transformation qualifies ONLY when the transformation itself is a clear conceptual mechanism
derived from the selected advantage and slogan.
""".strip()

BUILDER1_PERCEPTION_FIRST = """
PERCEPTION FIRST — before proposing visual objects, state internally what the viewer must acquire.
Why: objects alone do not persuade; a specific belief or clarity must change first.
Failure prevented: decorative scenes that look attractive but do not demonstrate the relative advantage.
Instead: define what should become newly clear, what physical law represents that belief, and which familiar object communicates it most immediately — clearer than showing the advertised product.
Selection test: "What should the viewer believe after seeing the advertisement?"
Product Name and fixed slogan connect the visual to the advertiser; the product itself does not need to carry that burden.
""".strip()

BUILDER1_TRANSFERRED_PHYSICAL_GENERATOR = """
TRANSFERRED PHYSICAL GENERATOR — preferred Builder1 physical generator:
- recognizable external object;
- visually simple;
- immediately understandable;
- capable of performing the slogan's implied action;
- surprising in the advertising context;
- repeatable across a coherent series;
- more useful for expressing the perception than the advertised product.

Methodology examples only (do not copy into unrelated campaigns — restart the physical search for every brand):
- To communicate that a city becomes shorter, show familiar long things that have become short — not merely a train.
- To communicate closeness, show a physical act of closeness — not merely the advertised service.
- To communicate generosity, show another object behaving generously — not merely a large portion of the product.
""".strip()

BUILDER1_REMOVAL_TEST = """
REMOVAL TEST — mandatory self-test before selecting the physical or conceptual generator.
Why: merely showing what is being sold usually creates no new perception and is transferable to competitors.
Failure prevented: conventional product shots where the idea collapses if the product is removed.
Instead: identify the exact perception and choose the clearest physical embodiment of that perception.
Selection test: "If the advertised product were removed from the visual, would a clear, persuasive, and distinctive advertising idea still remain?"
YES: real independent visual idea. NO: reject or redevelop — do not treat product presence as default proof.
""".strip()

BUILDER1_CLARITY_OVER_CATEGORY = """
CLARITY OVER CATEGORY LITERALNESS:
Why: category literalness feels obvious but often fails to demonstrate the relative advantage.
Failure prevented: choosing the product category object because it "matches" the brief without proving the perception.
Instead: prefer the object that explains the intended perception most clearly — even from another physical world when the analogy is immediate.
Selection test: "Does this object make the intended perception understandable in seconds, regardless of category?"
Do not reward category literalness by itself.
""".strip()

BUILDER1_DISTINCTIVENESS = """
DISTINCTIVENESS:
A product shot is often transferable to any competitor.
The selected generator must create an advertising world ownable by the specific brand.

Ask during evaluation:
- Could a competitor replace the Product Name and use the same execution?
- Does the physical mechanism express the selected relative advantage?
- Does the slogan naturally complete the visual?
- Does the campaign create a recognizable recurring visual law?
- Is the idea more distinctive than simply presenting the product?

Attractive but generically transferable candidates must not win.
""".strip()

BUILDER1_PRODUCT_EVIDENCE_EXCEPTION = """
PRODUCT APPEARANCE IS AN EXCEPTION, NOT THE STARTING POINT:
Why: showing the product tells what is sold but rarely proves why this option is better.
Failure prevented: attractive product presentation mistaken for an advertising idea.
Instead: use an external transferred object unless a genuine physical property of the product is necessary evidence for the selected advantage.
Selection test: "Is the product required as proof of a specific physical property, or merely for recognition?"
When productEvidenceRequired is true, state in productEvidenceReason why external proof is insufficient and what mechanism the product performs.
Attractive presentation alone is not proof.
""".strip()

BUILDER1_VISIBILITY_POLICY_METHODOLOGY = """
RELATION TO SERVER VISIBILITY POLICY:
The server owns productVisibilityPolicy. Do not override it.

When policy=FORBIDDEN:
- use an external transferred object;
- the advertised product and packaging must not appear;
- Product Name may appear only as plain typography.

When an explicit user request permits secondary product visibility:
- product presence remains secondary;
- the transferred generator must still carry the idea;
- the product must not become the default main visual;
- no packaging logo or invented mark may appear.
""".strip()

BUILDER1_SERIES_TRANSFERRED_OBJECT_RULES = """
SERIES — preserve transferred-object logic in every ad.
Why: one strong transferred-object ad followed by product shots breaks the campaign law the viewer learned.
Failure prevented: product-shot fallbacks, packaging variations, or decorative changes with no conceptual development.
Instead: each ad is a distinct execution of the same conceptual, physical-family, and graphic generators.
Selection test: "Does this ad continue the same visual mechanism as the others, or restart with a different idea?"
""".strip()

BUILDER1_FORBIDDEN_PRODUCT_SHOT_LANGUAGE = """
Do not treat the image as catalog packaging photography, a centered goods display, premium goods beauty lighting,
packaging presentation, or conventional commercial goods photography.
Describe positively what IS shown: the transferred external object, its physical action, the visual perception,
the graphic system, and Product Name plus slogan as typography only.
""".strip()

CONCEPTUAL_PRODUCT_SHOT_REJECTION_CODES = frozenset(
    {
        "concept_conventional_product_shot",
        "concept_collapses_without_product",
        "concept_product_shot_bias",
        "concept_category_literal_only",
        "concept_decorative_presentation_only",
        "concept_no_transferred_object_path",
        "concept_not_distinctive",
        "concept_starts_from_product_shape",
    }
)

PHYSICAL_PRODUCT_SHOT_REJECTION_CODES = frozenset(
    {
        "physical_conventional_product_shot",
        "physical_collapses_without_product",
        "physical_all_candidates_same_world",
        "physical_all_candidates_are_product",
        "physical_decorative_presentation_only",
        "physical_no_external_object",
        "physical_insufficient_candidates",
        "physical_missing_evidence_reason",
    }
)
