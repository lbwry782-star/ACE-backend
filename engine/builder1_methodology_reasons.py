"""
Builder1 methodology — reason-backed creative rules.

Each major rule explains why it exists, what failure it prevents, what to pursue
instead, and a practical selection test. Blocks are stage-specific excerpts from
this shared source to avoid duplicating full essays in every call.
"""

# --- Strategy stage ---

STRATEGY_PROBLEM_PERCEPTION = """
Begin by identifying the real business, perceptual, or customer problem — not generic praise.
Why: an advertisement must change a specific belief, hesitation, misunderstanding, or lack of association.
Failure prevented: generic statements such as high quality, excellent service, innovative, reliable, or perfect for you.
Instead: identify what the audience currently believes, fails to believe, prefers, forgets, misunderstands, or does not associate with the brand.
Selection test: "What audience perception must change for this advertisement to succeed?"
""".strip()

STRATEGY_RELATIVE_ADVANTAGE = """
Choose the relative advantage before any slogan or visual idea.
Why: a slogan and visual mechanism persuade only when they express a reason to choose this product over the realistic alternative.
Failure prevented: clever wording, decorative visuals, exaggerated claims, or generic emotional language compensating for no real difference.
Instead: find the most useful truthful difference from the customer's point of view. The advantage need not be absolute superiority; a real limitation may sometimes be reframed honestly. Ground it in the supplied brief — do not invent unsupported capabilities.
Selection test: "Why should the audience choose this option rather than the realistic alternative?"
""".strip()

STRATEGY_TRUTH_AND_OPERATIONS = """
The campaign must not depend on business changes that do not currently exist.
Why: Builder1 is a digital advertising agent — it can communicate a real advantage but cannot change pricing, service, inventory, hours, delivery policy, or operating model.
Failure prevented: an attractive campaign that becomes true only if the business later implements a new service or investment.
Instead: build from an advantage already true or honestly stated from the supplied information. At most one optional immediate negligible-cost communication action may be allowed when supported — campaign truth must not depend on it.
Selection test: "Would this advertisement remain truthful if the business made no operational change after receiving it?"
""".strip()

STRATEGY_STAGE_METHODOLOGY = "\n\n".join(
    [STRATEGY_PROBLEM_PERCEPTION, STRATEGY_RELATIVE_ADVANTAGE, STRATEGY_TRUTH_AND_OPERATIONS]
)

# --- Slogan stage ---

SLOGAN_AFTER_ADVANTAGE = """
Generate the brand slogan only after selecting the relative advantage.
Why: a strong slogan is usually the shortest memorable linguistic form of that advantage — not an independent clever sentence searching for strategy afterward.
Failure prevented: generic inspiration, empty cleverness, category clichés, rhymes unrelated to the problem, or slogans any competitor could use.
Instead: distill the selected relative advantage into a concise, memorable, ownable statement. Do not force clever copy when the advantage naturally yields a strong phrase. Uniqueness comes from wording and strategic idea — not from saying "we are unique."
Selection test: "If the relative advantage were removed, would this slogan still have the same meaning?" If yes, it may be too generic.
""".strip()

SLOGAN_FIXED_BEFORE_CONCEPT = """
The selected slogan becomes fixed before conceptual generation.
Why: the conceptual generator should be a visual interpretation of one strategic linguistic decision — not a moving target while visuals develop.
Failure prevented: several unrelated ads with a slogan invented afterward to loosely connect them.
Instead: let one fixed slogan generate one clear action, one conceptual law, and one campaign world.
Selection test: "Does this concept feel like a natural visual consequence of the fixed slogan, or was the slogan attached afterward?"
""".strip()

SLOGAN_IMPLIED_ACTION_INTRO = """
Extract the action implied by the slogan before selecting visual objects.
Why: objects alone do not create a campaign mechanism — a recurring action, transformation, or physical law can generate multiple distinct ads.
Failure prevented: choosing an arbitrary object that merely resembles a word in the slogan.
Instead: translate the slogan into something that can physically happen (shorten, connect, protect, reveal, bring closer, multiply, remove, open, support, carry, simplify — action types, not objects to copy).
Selection test: "What visible physical change or relationship performs the slogan rather than merely illustrating one of its nouns?"
""".strip()

SLOGAN_STAGE_METHODOLOGY = "\n\n".join(
    [SLOGAN_AFTER_ADVANTAGE, SLOGAN_FIXED_BEFORE_CONCEPT, SLOGAN_IMPLIED_ACTION_INTRO]
)

# --- Conceptual stage ---

CONCEPTUAL_GENERATOR_LAYER = """
Define the conceptual generator before choosing the physical object.
Why: the conceptual generator is the repeatable law behind the campaign; the physical object is only one embodiment.
Failure prevented: a one-off visual with strategic explanation retrofitted afterward.
Instead: define the reusable conceptual action first, then search for the clearest physical embodiment.
Layers: conceptual generator = what repeatedly happens; physical generator = what object performs it; graphic generator = how the campaign looks. Do not collapse these into one field.
Selection test: "Can this conceptual law generate several distinct advertisements without changing its central meaning?"
""".strip()

IDEA_BEFORE_OBJECT = """
Do not begin from an attractive object and ask what it could mean.
Why: object-first thinking produces decorative symbolism instead of persuasive advertising.
Failure prevented: hearts, keys, bridges, light bulbs, puzzle pieces, gifts, crowns, or rockets chosen merely because they look "creative."
Instead: begin with strategic perception and implied action; choose an object only because its physical behavior explains the idea clearly.
Selection test: "Was this object selected because of the exact physical role it performs, or because it is a familiar advertising symbol?"
Do not blacklist objects globally — the same object may be valid when its physical role is genuinely needed.
""".strip()

CLARITY_BEFORE_CLEVERNESS = """
Prefer the clearest strong embodiment over a more complicated clever one.
Why: an advertising image has very little time; a brilliant explanation afterward does not rescue an unclear visual.
Failure prevented: multi-step symbolism, obscure references, several simultaneous metaphors, or visuals that only make sense after reading internal rationale.
Instead: one dominant object, one dominant action, one dominant perception.
Selection test: "Could a viewer understand the central visual mechanism in a few seconds without reading the planning report?"
Clarity does not mean banality — seek a surprising idea with an immediate bridge.
""".strip()

DISTINCTIVENESS_AND_OWNERSHIP = """
The series must be ownable by the specific brand, not merely coherent.
Why: a visually attractive campaign has limited value when a competitor can replace the Product Name and use it unchanged.
Failure prevented: generic campaigns about quality, love, innovation, speed, freshness, trust, or togetherness without tying the mechanism to this advantage.
Instead: create a recurring visual law that feels like a natural consequence of this brand's specific advantage and slogan.
Selection test: "Could a realistic competitor replace the Product Name and use the same campaign without changing the idea?"
Do not repair generic campaigns by adding product cues afterward — restart the conceptual and physical search.
""".strip()

RESTART_NOT_RETROFIT = """
When the generator is strategically wrong or generic, do not patch it with brand or product details.
Why: a product cue can identify the advertiser but cannot transform a generic idea into an ownable campaign.
Failure prevented: reusing an earlier visual world and adding product color, package, logo-like symbol, shape, category prop, or extra headline.
Instead: return to strategic problem, advantage, slogan action, conceptual generator, and physical generator.
Selection test: "Was this visual designed from the current brand's advantage, or adapted from an idea that would work for someone else?"
""".strip()

CONCEPTUAL_STAGE_METHODOLOGY = "\n\n".join(
    [
        CONCEPTUAL_GENERATOR_LAYER,
        IDEA_BEFORE_OBJECT,
        CLARITY_BEFORE_CLEVERNESS,
        DISTINCTIVENESS_AND_OWNERSHIP,
        RESTART_NOT_RETROFIT,
    ]
)

# --- Brand physical stage ---

PRODUCT_SHOT_BIAS_REASON = """
Do not use the advertised product as the default conceptual or physical generator.
Why: showing the product tells what is being sold but often creates no new belief, does not demonstrate the relative advantage, and is transferable to competitors.
Failure prevented: centered product, enlarged product, premium lighting, person holding product, decorative packaging, multiplied or recolored product, or product in a dramatic environment.
Instead: find the clearest physical demonstration of the perception, even from a completely different visual world.
Selection test: "If the advertised product were removed, would a clear persuasive visual idea still remain?"
Exception: the product may participate only when a genuine physical property is necessary proof of the selected advantage — attractive presentation alone is not proof.
""".strip()

TRANSFERRED_OBJECT_REASON = """
Prefer a recognizable external object that performs the slogan's implied action.
Why: a familiar object lets the viewer understand an abstract advantage through a physical law they already know.
Failure prevented: an unrelated surreal object chosen merely because it is surprising.
Instead: choose an object whose familiar identity makes the intended change immediately understandable. Surprise without clarity is not enough; relevance without surprise may be banal; the strongest object combines familiarity, clarity, and unexpected use. It need not belong to the product category.
Selection test: "Does the viewer understand why this particular object is doing this particular action, or is the connection dependent on an explanation?"
""".strip()

DO_NOT_SHOW_THING_ITSELF = """
Do not literally depict the subject of the slogan when another object can communicate the idea more clearly.
Why: literal depiction repeats the claim; a transferred embodiment demonstrates it.
Failure prevented: showing the service interface for closeness, the vehicle for speed, or the food product for freshness — category illustration without perception proof.
Instead: find a known physical situation where closeness, speed, freshness, generosity, simplicity, or protection becomes visibly undeniable.
Selection test: "Is the visual merely showing the subject, or is it physically proving the intended perception?"
Do not copy these categories automatically into campaigns.
""".strip()

METHODOLOGY_EXAMPLE_TEACHING = """
When using methodology examples, infer the transferable mechanism — do not copy the visible object.
Why: without explanation, models copy the object rather than the principle.
Failure prevented: a short-necked giraffe example causing giraffes in unrelated campaigns.
Example teaching: "A giraffe works because it is universally recognized for its long neck. Shortening that known long feature demonstrates 'becoming shorter.' The principle is shortening something familiarly long — not using a giraffe."
Mark all examples as methodology-only; restart the physical search for every brand.
""".strip()

BRAND_PHYSICAL_STAGE_METHODOLOGY = "\n\n".join(
    [
        PRODUCT_SHOT_BIAS_REASON,
        TRANSFERRED_OBJECT_REASON,
        DO_NOT_SHOW_THING_ITSELF,
        METHODOLOGY_EXAMPLE_TEACHING,
    ]
)

# --- Graphic system stage ---

GRAPHIC_GENERATOR_REASON = """
Define the graphic generator before producing individual ads.
Why: a series is recognized by a stable visual language, not only by its idea.
Failure prevented: ads with the same message but different palettes, typography, composition, image styles, camera treatments, realism levels, or layouts — several unrelated advertisements.
Instead: one consistent visual system making every execution immediately recognizable as part of the same campaign.
Selection test: "If the Product Name were temporarily hidden, would these advertisements still look like members of one series?"
Define palette, typography behavior, composition grid, image style, object scale, text placement, background treatment, and recurring visual relationships before ads exist.
""".strip()

# --- Series ads stage ---

SERIES_COHERENCE_REASON = """
Every ad must use the same conceptual, physical-family, and graphic generators.
Why: a series develops one idea through variation — not a collection of different metaphors about the same broad benefit.
Failure prevented: treating "connection" as permission for handshake, bridge, chain, cable, hug, and puzzle pieces in one series — similar message, unstable visual family.
Instead: choose one recurring visual law and develop distinct examples within it.
Selection test: "Do these advertisements feel like different episodes of one mechanism, or different ideas gathered under one topic?"
""".strip()

VARIATION_NOT_REPETITION = """
Each advertisement should be part of the series but contribute a new execution.
Why: exact repetition wastes a series; excessive variation destroys coherence.
Failure prevented: repeating the same object with a different background, or changing the generator entirely in every ad.
Instead: keep the recurring law fixed while changing the specific example, setting, subject, scale, or physical instance.
Selection test: "What remains constant, and what meaningfully changes?" — identify both.
""".strip()

HEADLINE_OPTIONAL_REASON = """
Do not add a headline automatically.
Why: a strong visual may already communicate the idea; extra copy can explain the joke twice, clutter the composition, or compete with the brand slogan.
Failure prevented: treating every ad as requiring Product Name, headline, slogan, and supporting text inside the image.
Instead: use a headline only when the visual and fixed slogan do not communicate the intended idea clearly enough.
Selection test: "Would removing the headline make the intended perception unclear?" If no, omit it.
When used: maximum seven words; Product Name excluded from that count; headline must differ from the fixed brand slogan.
""".strip()

SLOGAN_VS_HEADLINE_REASON = """
The campaign slogan remains fixed; an optional headline may vary by ad.
Why: the slogan is the recurring strategic promise; the headline, when needed, helps interpret one specific execution.
Failure prevented: inventing a new slogan per ad or repeating the same information in both fields.
Instead: slogan = permanent campaign signature; headline = local aid for one visual only.
Selection test: "Is this line expressing the permanent brand promise, or helping explain this particular visual?"
""".strip()

SERIES_STAGE_METHODOLOGY = "\n\n".join(
    [
        SERIES_COHERENCE_REASON,
        VARIATION_NOT_REPETITION,
        HEADLINE_OPTIONAL_REASON,
        SLOGAN_VS_HEADLINE_REASON,
        RESTART_NOT_RETROFIT,
    ]
)

# --- Image prompt and compliance ---

NO_LOGO_REASON = """
Do not invent or depict logos, marks, emblems, badges, seals, monograms, or brand-like symbols.
Why: the system has no approved brand asset; an invented mark is misleading, unstable across the series, and may appear unauthorized.
Failure prevented: initials, symbols, package marks, decorative seals, fake trademarks, or emblematic icons near the Product Name.
Instead: identify the brand through plain readable Product Name typography and the fixed campaign graphic system.
Selection test: "Is this element ordinary advertising typography, or could a viewer interpret it as an official brand symbol?"
A legitimate object in the visual idea is not a logo merely because it is visually distinctive.
""".strip()

POSITIVE_IMAGE_PROMPT_REASON = """
State clearly what should be shown — not only what must be absent.
Why: a long negative list leaves the image model without a strong central subject; it may reintroduce the forbidden object as the most concrete subject.
Failure prevented: prompts listing no product, no package, no logo, no product shot — but giving only a vague transferred object.
Instead: specify one unmistakable main visual, one physical action, one composition, and one graphic system.
Selection test: "If all negative instructions were removed, would the positive prompt still describe a complete, specific image?"
""".strip()

IMAGE_COMPLIANCE_REASON = """
Image compliance checks enforce identity and visibility rules structurally — not creative taste.
Why: invented logos and unauthorized product depiction undermine campaign truth and brand safety regardless of visual quality.
Failure prevented: the image model adding symbols, packaging marks, or product hero shots when policy forbids them.
Instead: plain Product Name typography, campaign graphic devices as composition only, and the planned transferred object as main visual when product visibility is forbidden.
""".strip()

# Stage bundles for token estimation and tests
STAGE_METHODOLOGY_BLOCKS: dict[str, str] = {
    "strategy_stage": STRATEGY_STAGE_METHODOLOGY,
    "strategy_slogan_stage": STRATEGY_STAGE_METHODOLOGY + "\n\n" + SLOGAN_STAGE_METHODOLOGY,
    "slogan_stage": SLOGAN_STAGE_METHODOLOGY,
    "conceptual_stage": CONCEPTUAL_STAGE_METHODOLOGY,
    "brand_physical": BRAND_PHYSICAL_STAGE_METHODOLOGY,
    "graphic_system": GRAPHIC_GENERATOR_REASON,
    "series_ads": SERIES_STAGE_METHODOLOGY,
    "image_prompt": POSITIVE_IMAGE_PROMPT_REASON,
}
