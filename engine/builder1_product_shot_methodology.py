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
Instead: prefer the object that explains the intended perception most clearly — product/category first when it carries
a genuine advertising mechanism; otherwise an external object when it adds a capability the direct route cannot match.
Selection test: "Does this object make the intended perception understandable in seconds, regardless of category?"
Do not reward category literalness by itself — but do not reject a direct category route merely because an external analogy also exists.
""".strip()

BUILDER1_POPULAR_ANALOGY_FIRST = """
POPULAR ANALOGY FIRST — scope: ONLY after ANALOGY_LED is already justified by the direct-product pre-route gate.
This rule does NOT decide PRODUCT_LED vs ANALOGY_LED.

Once external analogy is justified and ANALOGY_LED is the chosen route:
When several physical analogies can express the same relative advantage, prefer the most widely understood
object, action, cause/effect, or everyday situation — not the most technically precise or professionally elegant mechanism.
Why: advertising clarity beats engineering sophistication.
Failure prevented: autofocus loops, industrial inspection lines, or optical systems that ordinary viewers cannot read instantly.
Instead: magnet attracts, door opens, umbrella blocks rain, domino falls, key opens lock — familiar mechanisms with immediate causal readability.
Selection test: "Could a reasonably observant child describe what is physically happening in one simple sentence?"
Do NOT hard-ban technical objects — use them only when instantly obvious without specialist knowledge.
Never choose the popular object first and retrofit meaning afterward; derive it from the relative advantage.
Never select an external analogy merely because a familiar popular analogy exists.
""".strip()

BUILDER1_PUBLIC_SIMPLICITY = """
PUBLIC SIMPLICITY (child-comprehension heuristic — not childish art direction):
Treat the viewer as general public without assuming technical, professional, academic, or industrial knowledge.
The idea may be sophisticated; the physical action must be simple.
Every planned execution must satisfy two sentences:
1) What is physically happening? — one short ordinary-language sentence.
2) Why does that express the relative advantage? — one short ordinary-language sentence.
If the explanation needs specialist vocabulary, invisible machinery, or three+ hidden symbolic mappings, reject or repair.
immediateClarityReason must prove: what the viewer sees happen, that ordinary viewers understand that action,
and how that action connects directly to the relative advantage — not merely that an effect or object is familiar.
Do NOT prefer mechanisms because they sound engineered, systematic, scientific, or professional.

Public Simplicity means the audience decodes the ad simply — it does NOT mean "use a simple everyday analogy."
When the product/category is already simple and readable, replacing it with another simple domain (food, tray, etc.)
does not improve Public Simplicity unless the transfer creates a clearly stronger advertising mechanism.
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
PRODUCT / DIRECT ROUTE MUST BE TESTED BEFORE EXTERNAL ANALOGY:
Why: when the product or category is simple and the relative advantage can be shown directly, forcing an external
domain adds translation cost without persuasive gain.
Failure prevented: attractive product presentation mistaken for an advertising idea — AND unnecessary cross-domain
analogy when the product/category itself can carry a genuine advertising mechanism.
Instead: first test whether a PRODUCT_LED or PRODUCT_INTEGRATED_ANALOGY mechanism exists; only then consider external
transferred objects when they add a specific capability the direct route cannot provide equally well.
Selection test: "Can the product/category demonstrate the advantage through a real advertising mechanism — not merely recognition?"
When productEvidenceRequired is true, state in productEvidenceReason why external proof is insufficient and what mechanism the product performs.
Attractive presentation alone is not proof — but a meaningful product-led transformation is valid advertising.
""".strip()

BUILDER1_VISIBILITY_POLICY_METHODOLOGY = """
RELATION TO SERVER VISIBILITY POLICY:
The server owns productVisibilityPolicy. Do not override it.

When policy=CREATIVE_DECISION (default):
- There is no necessity to show the product — but showing it may be the strongest idea.
- Choose AFTER strategy, relative advantage, and slogan are fixed.
- Three valid routes: ANALOGY_LED (external transferred object), PRODUCT_LED (product itself carries the idea),
  PRODUCT_INTEGRATED_ANALOGY (product participates in a larger mechanism).
- Prefer whichever route most clearly expresses the relative advantage with public simplicity.
- Do NOT manufacture an external analogy merely to avoid showing a simple everyday product.
- Do NOT show the product merely because no better idea was found — product-led requires a genuine creative mechanism.
- Product Name may appear only as plain typography; no invented logo or packaging brand mark.

When policy=PRODUCT_VISIBILITY_REQUIRED (explicit user show-product request):
- The product must appear in the image — presence is mandatory.
- Visual hierarchy is NOT predetermined: choose the strongest route after strategy.
- Valid routes remain PRODUCT_LED, PRODUCT_INTEGRATED_ANALOGY, or ANALOGY_LED with visible product.
- Do NOT default to a generic packshot; product-led still requires a genuine creative mechanism.
- Do NOT force secondary placement when product-led or integrated analogy is stronger.

When policy=FORBIDDEN (explicit user hide-product request or legacy stored campaigns):
- use an external transferred object;
- the advertised product and packaging must not appear;
- Product Name may appear only as plain typography.

When policy=SECONDARY_EXPLICIT_EXCEPTION (legacy stored campaigns only):
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
        "concept_literal_slogan_illustration",
        "concept_slogan_noun_depiction",
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
        "physical_literal_slogan_object",
        "physical_slogan_word_illustration",
        "physical_route_assessment_missing",
        "physical_route_assessment_inconsistent",
        "physical_analogy_without_unique_gain",
        "physical_unjustified_external_analogy",
    }
)
