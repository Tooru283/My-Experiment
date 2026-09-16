"""Two bounded text extraction stages; preserve explicit final stopping clauses.

These checks enforce source-text preservation, not general semantic correctness.
No execution, progress or visual evidence is inferred here.
"""
import json
import re


ACTION_SYSTEM = """Extract the ordered intended actions of a navigation instruction.
Return only JSON: {"actions":["action phrase", ...]}.
Copy original wording, spatial relations and identifying modifiers. Starting
location descriptions are context, not movement commands. Split independent
sequential actions, but keep compound or conditional actions together when a
split would lose their meaning: 'Turn right to walk past stairs' and 'Walk forward
until you reach the sofa' each remain one action. Seeing a landmark is not proof
of executing an action. Do not output landmarks or completion judgements.
Repeat an elided route verb when this is needed to make state changes atomic:
'Leave the bathroom and closet' becomes 'Leave the bathroom', 'Leave the closet';
'Go toward the house and through the sliding doors' becomes 'Go toward the house',
'Go through the sliding doors'.
Preserve every stop/wait directive once, including its conditions, modifiers and
reference objects. A pause followed by later movement is intermediate, not the final
destination. Keep repeated words such as "stop, and wait" in one action. Never create
two STOP actions from one source STOP.

Instruction: Go through the door and turn left. Stop in the doorway to the left of the white double doors.
Output: {"actions":["Go through the door","turn left","Stop in the doorway to the left of the white double doors"]}
Instruction: Go into the living area with the fireplace. Stop in the area between the two white sofas, next to the entrance to the dining room.
Output: {"actions":["Go into the living area with the fireplace","Stop in the area between the two white sofas, next to the entrance to the dining room"]}
Instruction: You are in a closet. Turn right to walk out of the closet into a bedroom. Walk forward until you reach the sofa.
Output: {"actions":["Turn right to walk out of the closet into a bedroom","Walk forward until you reach the sofa"]}
"""

LANDMARK_SYSTEM = """Extract physical landmarks from the ORIGINAL instruction and
the supplied intended actions. Return only JSON with these two keys:
{"landmarks":["phrase", ...],"terminal_target":"primary stopping place"}.
Use JSON null for terminal_target if no final stop/wait directive exists. A stop
followed by a later movement is intermediate and does not define terminal_target.
Landmarks are objects or regions; preserve source wording and modifiers. Do not
invent observations or report completion. The original instruction is authoritative.

Distinguish the primary stopping place from its reference objects. Copy the primary
place as terminal_target without the stop verb, leading location preposition or
leading article. Keep modifiers such as 'on right', 'to the room', 'at the top of
the stairs', and joint targets such as 'tub and sink'. The primary place ends before
a comma or a reference clause 'next to', 'to the left of', 'to the right of'. Keep
these reference objects separately in landmarks. For a BETWEEN destination, keep
the whole 'area between the two white sofas' phrase as the primary place.
Put route landmarks first, terminal reference objects next, and terminal_target
last. Both the stopping place and its references must be included. Resolve terminal deixis and event conditions to their physical place: "wait there"
uses the preceding destination, and "Stop once you enter the next room" targets the
next room rather than the words "once you enter". If the final STOP is only a safety
condition such as "Stop before going outside", use the immediately preceding route
destination; do not turn "before going outside" into a landmark or target.

Instruction: Go to the outdoor foyer. Stop before going outside.
Output: {"landmarks":["outdoor foyer"],"terminal_target":"outdoor foyer"}
Instruction: Stop in the doorway to the left of the white double doors.
Output: {"landmarks":["white double doors","doorway"],"terminal_target":"doorway"}
Instruction: Leave the bedroom. Stop in the doorway to the room.
Output: {"landmarks":["bedroom","room","doorway to the room"],"terminal_target":"doorway to the room"}
Instruction: Go into the living area with the fireplace. Stop in the area between the two white sofas, next to the entrance to the dining room.
Output: {"landmarks":["living area with the fireplace","two white sofas","entrance to the dining room","area between the two white sofas"],"terminal_target":"area between the two white sofas"}
Instruction: Walk into bathroom and wait near the tub and sink.
Output: {"landmarks":["bathroom","tub","sink","tub and sink"],"terminal_target":"tub and sink"}
"""

_STOP = re.compile(r"\b(?:stop|wait|stay|stand|pause)\b", re.I)
_LEADING_LOCATION = re.compile(
    r"^\s*(?:(?:right|just|directly)\s+)?"
    r"(?:(?:in front of|next to|outside of|inside of|in|at|on|by|near|beside|inside|outside|under)\s+)?"
    r"(?:(?:the|a|an)\s+)?", re.I,
)
# A safety condition does not name a new target. The preceding route destination
# remains authoritative, e.g. ``Go to the foyer. Stop before going outside``.
_TEMPORAL_STOP_CONDITION = re.compile(
    r"^\s*(?:before|after|until|while)\s+"
    r"(?:(?:you|we|the\s+agent|the\s+robot)\s+)?"
    r"(?:go(?:ing)?|walk(?:ing)?|move|moving|head(?:ing)?|proceed(?:ing)?|"
    r"enter(?:ing)?|exit(?:ing)?|leave|leaving|step(?:ping)?|cross(?:ing)?|"
    r"pass(?:ing)?)\b",
    re.I,
)
_SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.S)
_CONDITION_LEAD = re.compile(r"^\s*(?:when|once|after|before|until|while)\b", re.I)
_ANAPHORIC_LEAD = re.compile(
    r"^\s*(?:(?:that['’]?s|that is|this is)\s+)?where\b", re.I
)
_EVENT_DESTINATION = re.compile(
    r"\b(?:once|when|after|until)\s+"
    r"(?:(?:you|we|the\s+agent|the\s+robot)\s+)?"
    r"(?:get\s+(?:to|into|past)|enter|exit|leave|reach|pass)\s+"
    r"(?P<target>.+?)(?=,?\s+(?:and\s+)?(?:stop|wait|stay|stand|pause)\b|[.!?]|$)",
    re.I,
)
_INDEPENDENT_ROUTE_AFTER = re.compile(
    r"(?:^|[.!?]\s+|\bthen\s+)\s*"
    r"(?:following\s+that[,]?\s*)?"
    r"(?:turn|take|make|go|walk|move|head|enter|exit|leave|climb|descend|continue|proceed|cross)\b",
    re.I,
)
_ANAPHORIC_TARGET = re.compile(
    r"^(?:there|here|it|this|that|the\s+room|this\s+room|that\s+room|room)$", re.I
)


def _words(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _json_object(response, keys):
    raw = str(response or "").strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try:
        value = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("Expected a complete JSON object") from exc
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("Expected exactly these JSON keys: " + ", ".join(keys))
    return value


def _phrases(value, allow_empty=False):
    if (not isinstance(value, list) or len(value) > 64
            or (not value and not allow_empty)):
        raise ValueError("Expected an array of at most 64 nonempty phrases")
    if any(not isinstance(x, str) or not x.strip() or "\n" in x or "\r" in x for x in value):
        raise ValueError("Each phrase must be a nonempty single-line string")
    return [x.strip() for x in value]


def _terminal_spans(instruction):
    """Return source-grounded STOP/wait spans with their original offsets."""
    spans = []
    text = str(instruction or "")
    for sentence in _SENTENCE.finditer(text):
        raw = sentence.group(0)
        matches = list(_STOP.finditer(raw))
        if not matches:
            continue
        first = matches[0]
        prefix = raw[:first.start()]
        # A leading event condition owns the STOP semantics. Ordinary route text
        # before ``and stop`` is a separate action and is not part of the directive.
        local_start = 0 if (
            _CONDITION_LEAD.match(prefix)
            or re.search(r"\b(?:there|here|where)\b", prefix, re.I)
        ) else first.start()
        while local_start < len(raw) and raw[local_start].isspace():
            local_start += 1
        value = raw[local_start:].strip().rstrip(". ")
        if value:
            spans.append((sentence.start() + local_start, sentence.end(), value))
    return spans


def _terminal_span_groups(instruction):
    """Group directives separated by no later route action into one requirement."""
    text = str(instruction or "")
    groups = []
    for span in _terminal_spans(text):
        if not groups or _INDEPENDENT_ROUTE_AFTER.search(
            text[groups[-1][-1][1]:span[0]]
        ):
            groups.append([span])
        else:
            groups[-1].append(span)
    return groups


def _anaphoric_terminal_span(value):
    return bool(re.search(r"\b(?:there|here|where)\b", str(value or ""), re.I))


def _effective_terminal_span(instruction):
    """Choose the first terminal directive after the final route command.

    Repeated ``stop ... and wait`` or ``Stop X. Wait there`` describes one
    terminal region. A pause followed by a later movement remains intermediate.
    """
    text = str(instruction or "")
    eligible = []
    for start, end, value in _terminal_spans(text):
        if not _INDEPENDENT_ROUTE_AFTER.search(text[end:]):
            eligible.append((start, end, value))
    return eligible[0] if eligible else None


def terminal_clause(instruction):
    span = _effective_terminal_span(instruction)
    return span[2] if span else ""


def _terminal_tail(instruction):
    clause = terminal_clause(instruction)
    if not clause:
        return ""
    # Conditional-prefix forms are handled by _event_destination_target.
    if _CONDITION_LEAD.match(clause):
        return clause
    return _STOP.sub("", clause, count=1).strip(" ,")


def _clean_destination(value):
    target = str(value or "").strip(" ,;.!?")
    target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.I)
    target = re.split(r"\bwhich\s+(?:will|would|is|was)\b", target, maxsplit=1, flags=re.I)[0]
    target = re.sub(r"(?:[,;]\s*)?\b(?:and|then)\s*$", "", target, flags=re.I)
    return target.strip(" ,;.!?") or None


def _preceding_route_destination(instruction):
    """Return a source phrase from the final route clause before terminal STOP."""
    prefix = str(instruction or "").strip().rstrip(" ,;.!?")
    prefix = re.sub(r"(?:[,;]\s*)?\b(?:and|then)\s*$", "", prefix, flags=re.I)
    patterns = (
        # Prefer the final explicit prepositional destination. Commas prevent an
        # earlier relation such as ``door to the right`` from swallowing the rest.
        r"\b(?:to|toward|towards|into|through|at|by|near|beside|inside)\s+(?:the\s+|a\s+|an\s+)?(?P<target>[^,;.!?]+?)\s*$",
        r"\b(?:get(?:\s+to|\s+into)?|reach|take|enter|find|visit|approach)\s+(?:the\s+|a\s+|an\s+)?(?P<target>[^;.!?]+?)\s*$",
        r"\b(?:go|walk|move|head|proceed|continue|make|turn|be)\b[^;.!?]*?\b(?:to|toward|towards|into|in|at|by|near|beside|inside)\s+(?:the\s+|a\s+|an\s+)?(?P<target>[^;.!?]+?)\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, prefix, re.I)
        if not match:
            continue
        target = _clean_destination(match.group("target"))
        if target and not re.match(r"^(?:your|my|our|his|her|their)\b", target, re.I):
            return target
    return None


def _event_destination_target(clause, instruction, start):
    match = _EVENT_DESTINATION.search(str(clause or ""))
    if not match:
        return None
    target = _clean_destination(match.group("target"))
    if target and _ANAPHORIC_TARGET.fullmatch(target):
        target = None
    return target or _preceding_route_destination(str(instruction or "")[:start])


def _temporal_stop_target(instruction):
    """Bind a safety-condition STOP to the immediately preceding destination."""
    span = _effective_terminal_span(instruction)
    if not span or not _TEMPORAL_STOP_CONDITION.match(_terminal_tail(instruction)):
        return None
    return _preceding_route_destination(str(instruction or "")[:span[0]])


def primary_terminal_place(instruction):
    span = _effective_terminal_span(instruction)
    if not span:
        return None
    start, _, clause = span
    tail = _terminal_tail(instruction)
    if _TEMPORAL_STOP_CONDITION.match(tail):
        return _preceding_route_destination(str(instruction or "")[:start])
    event_target = _event_destination_target(clause, instruction, start)
    if event_target:
        return event_target
    if re.fullmatch(r"(?:right\s+)?(?:there|here)", tail, re.I):
        return _preceding_route_destination(str(instruction or "")[:start])
    tail = _LEADING_LOCATION.sub("", tail, count=1)
    primary = re.split(
        r",|\b(?:next to|to the left of|to the right of|in front of|near)\b|"
        r"\b(?:before|after)\s+(?:you|we|the\s+agent|the\s+robot)\b",
        tail, maxsplit=1, flags=re.I,
    )[0]
    return primary.strip().rstrip(". ") or None


def _contains_word_sequence(haystack, needle):
    if not needle:
        return True
    width = len(needle)
    return any(haystack[i:i + width] == needle for i in range(len(haystack) - width + 1))

def _atomic_route_actions(action):
    """Split only unambiguous coordinated route state changes."""
    if _STOP.match(action) or re.search(r"\b(?:until|while|after|before)\b", action, re.I):
        return [action]
    parts = re.split(
        r",?\s+and\s+(?=(?:turn|take|go|walk|enter|exit|leave|climb|descend)\b)",
        action, flags=re.I,
    )
    expanded = []
    for part in parts:
        leave = re.match(
            r"^\s*(leave|exit)\s+(the\s+)?([^,;]+?)\s+and\s+(?:the\s+)?"
            r"([^,;]+?)\s*$", part, re.I,
        )
        approach_cross = re.match(
            r"^\s*((?:go|walk|move|head|proceed)\s+toward\s+.+?)\s+and\s+"
            r"(through|into)\s+(.+?)\s*$", part, re.I,
        )
        if leave and not re.search(
            r"\b(?:turn|take|go|walk|enter|exit|leave|climb|descend)\b",
            leave.group(4), re.I,
        ):
            verb = leave.group(1)
            article = leave.group(2) or ""
            expanded.extend([
                "{} {}{}".format(verb, article, leave.group(3)).strip(),
                "{} {}{}".format(verb, article, leave.group(4)).strip(),
            ])
        elif approach_cross:
            verb = re.match(r"\s*([a-z]+)", approach_cross.group(1), re.I).group(1)
            expanded.extend([
                approach_cross.group(1).strip(),
                "{} {} {}".format(verb, approach_cross.group(2), approach_cross.group(3)).strip(),
            ])
        else:
            expanded.append(part.strip())
    return expanded


def parse_actions(response, instruction):
    actions = _phrases(_json_object(response, ("actions",))["actions"])
    source_groups = _terminal_span_groups(instruction)
    source_spans = [span for group in source_groups for span in group]
    output_stop_actions = sum(bool(_STOP.search(action)) for action in actions)
    # Repeated ``stop ... wait there`` wording may collapse to one action. A single
    # source span may never expand into two STOP actions (the EP513 regression).
    if not (len(source_groups) <= output_stop_actions <= len(source_spans)):
        raise ValueError("STOP/wait count changed; preserve each original stopping requirement")

    # Validate each physical/source-qualified STOP span. A later anaphoric repeat
    # such as ``That's where you will wait`` may be omitted after the place-bearing
    # directive has already been preserved.
    output_words = _words(" ".join(actions))
    for group in source_groups:
        for index, (_, _, source_span) in enumerate(group):
            required = _words(source_span)
            if _contains_word_sequence(output_words, required):
                continue
            if index > 0 and _anaphoric_terminal_span(source_span):
                continue
            raise ValueError("STOP/wait clause lost source words: " + source_span)

    normalized = []
    for action in actions:
        match = _STOP.search(action)
        prefix = action[:match.start()] if match else ""
        owned_terminal_prefix = bool(
            prefix and (
                _CONDITION_LEAD.match(prefix) or _ANAPHORIC_LEAD.match(prefix)
            )
        )
        if match and prefix.strip(" ,;") and not owned_terminal_prefix:
            # Separate route execution from a following pause. This applies to
            # intermediate pauses too; their position in the action list remains
            # intact and build_anchor_chain decides whether they are terminal.
            route = re.sub(
                r"(?:[,;]\s*)?\b(?:and|then)\s*$", "", prefix,
                flags=re.I,
            ).strip(" ,;")
            if route:
                normalized.extend(_atomic_route_actions(route))
            normalized.append(action[match.start():].strip())
        elif owned_terminal_prefix:
            normalized.append(action.strip())
        else:
            normalized.extend(_atomic_route_actions(action))
    return normalized


def parse_landmarks(response, instruction):
    result = _json_object(response, ("landmarks", "terminal_target"))
    landmarks = _phrases(result["landmarks"], allow_empty=True)
    model_target = result["terminal_target"]
    if model_target is not None and (
        not isinstance(model_target, str) or not model_target.strip()
        or "\n" in model_target or "\r" in model_target
    ):
        raise ValueError("terminal_target must be a nonempty single-line string or null")

    expected = primary_terminal_place(instruction)
    span = _effective_terminal_span(instruction)
    if expected is None:
        if model_target is not None and (
            span is not None or not _STOP.search(str(instruction or ""))
        ):
            raise ValueError("No named explicit stopping place: terminal_target must be null")
        # An explicit pause followed by more route movement is intermediate. Its
        # object may remain a route landmark, but it must not become final authority.
        return landmarks

    clause = span[2] if span else ""
    tail = _terminal_tail(instruction)
    semantic_condition = bool(
        _CONDITION_LEAD.match(clause)
        or _EVENT_DESTINATION.search(clause)
        or _TEMPORAL_STOP_CONDITION.match(tail)
        or re.fullmatch(r"(?:right\s+)?(?:there|here)", tail, re.I)
    )
    full_place = expected if semantic_condition else _LEADING_LOCATION.sub(
        "", tail, count=1
    )

    # The source-derived primary target is authoritative. The model supplies the
    # landmark inventory; deterministic normalization corrects reference-object
    # selection, truncated target modifiers, and harmless ordering mistakes.
    verbose_reference = None
    if isinstance(model_target, str):
        model_target = model_target.strip()
        ew, mw = _words(expected), _words(model_target)
        if mw != ew and len(mw) > len(ew) and mw[:len(ew)] == ew:
            remainder = model_target[len(expected):].strip(" ,")
            remainder = re.sub(
                r"^(?:next to|to the left of|to the right of|in front of|near)\s+",
                "", remainder, flags=re.I,
            )
            verbose_reference = re.sub(
                r"^(?:the|a|an)\s+", "", remainder, flags=re.I,
            ).strip() or None

    normalized = []
    seen = set()
    for item in landmarks:
        words = tuple(_words(item))
        if words == tuple(_words(expected)):
            continue
        if isinstance(model_target, str) and words == tuple(_words(model_target))                 and tuple(_words(model_target)) != tuple(_words(expected))                 and verbose_reference:
            continue
        if words and words not in seen:
            normalized.append(item)
            seen.add(words)
    if verbose_reference:
        words = tuple(_words(verbose_reference))
        if words and words not in seen:
            normalized.append(verbose_reference)
            seen.add(words)
    normalized.append(expected)
    landmarks = normalized

    # Require all concrete terminal nouns/modifiers to survive somewhere in the
    # inventory. Function words and event verbs are represented by the action,
    # rather than being miscast as landmark names.
    ignored = set(
        "stop wait stay stand pause the a an in at by near beside next to of on "
        "and or with left right front inside outside between area your you we this that "
        "once when after before until while get go going walk walking move moving "
        "head heading proceed proceeding enter entering exit exiting leave leaving "
        "step stepping cross crossing pass passing reach then will be where".split()
    )
    required_source = expected if semantic_condition else clause
    required = set(_words(required_source)) - ignored
    missing = required - set(_words(" ".join(landmarks)))
    if missing:
        raise ValueError(
            "Terminal reference/modifier words missing from landmarks: "
            + ", ".join(sorted(missing))
        )
    return landmarks
