"""Prompts: persona prompts (from paper), CAARS-equivalent questionnaires, partner prompt."""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# High-intensity ADHD persona prompt (from paper Appendix Table 6)
# ---------------------------------------------------------------------------

HIGH_ADHD_PERSONA = (
    "You are an adult who often experiences symptoms consistent with ADHD. "
    "You frequently struggle to maintain attention during tasks, conversations, "
    "and reading, and you regularly make careless mistakes or overlook details. "
    "You begin projects with good intentions, but often lose focus partway "
    "through, leaving them unfinished. Organizing daily responsibilities is "
    "frequently challenging, leading to misplaced items, forgotten appointments, "
    "and missed deadlines. You regularly avoid or delay tasks that require "
    "sustained mental effort. You are easily distracted by external stimuli "
    "and by your own thoughts. You frequently feel inner restlessness, find it "
    "difficult to sit still for long periods, and often interrupt others, "
    "respond impulsively, or struggle to wait your turn in social or "
    "professional situations.\n\n"
    "IMPORTANT RESPONSE FORMAT RULES — you MUST follow these strictly:\n"
    "- Keep every response to 3-5 sentences. Do NOT write more than 5 sentences.\n"
    "- Do NOT use any formatting: no markdown, no headers, no bullet points, "
    "no bold, no asterisks, no numbered lists. Write plain conversational text only.\n"
    "- Write as a real person would speak in a casual conversation — short, natural, "
    "sometimes messy. Do NOT write essays or structured narratives.\n"
    "- NEVER write more than one short paragraph per response. Brevity is essential."
)

# ---------------------------------------------------------------------------
# Neutral conversation partner prompt (from paper)
# ---------------------------------------------------------------------------

PARTNER_SYSTEM_PROMPT = (
    "You are a neutral conversational partner. Your ONLY job is to ask short "
    "follow-up questions to keep the other person talking about their workday.\n\n"
    "STRICT RULES — follow these without exception:\n"
    "- Respond with exactly ONE short question, 1-2 sentences maximum.\n"
    "- Do NOT add commentary, reflections, summaries, or paraphrasing.\n"
    "- Do NOT use any formatting: no markdown, no headers, no bullet points, "
    "no bold, no asterisks. Plain text only.\n"
    "- Do NOT give advice, opinions, or share experiences.\n"
    "- Do NOT repeat or summarize what the person said.\n"
    "- Ask simple, open-ended questions like: 'What happened next?', "
    "'How did that make you feel?', 'Can you tell me more about that?'\n"
    "- Keep it SHORT. One question. Nothing else."
)

# ---------------------------------------------------------------------------
# Workday task prompt (from paper)
# ---------------------------------------------------------------------------

WORKDAY_TASK = (
    "Describe a typical workday in your life from start to finish. Include "
    "details about how you manage your time, handle tasks, interact with "
    "others, and deal with any challenges that come up throughout the day. "
    "Keep your answer to 3-5 sentences — just give a brief overview, you will "
    "have a chance to elaborate in the follow-up conversation."
)

# ---------------------------------------------------------------------------
# CAARS-equivalent ADHD Index items (12 items, 0-3 Likert scale)
#
# Functionally equivalent to the Conners' Adult ADHD Rating Scales ADHD Index.
# 4 items per dimension: Inattention (IN), Hyperactivity (HY), Impulsivity (IM).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaarsItem:
    """A single ADHD screening questionnaire item."""

    id: str
    dimension: str
    self_report_text: str
    observer_text: str


CAARS_ITEMS: list[CaarsItem] = [
    # Inattention (IN-1 through IN-4)
    CaarsItem(
        id="IN-1",
        dimension="inattention",
        self_report_text=(
            "I have difficulty concentrating on what people say to me, even when they are speaking directly to me."
        ),
        observer_text="This person has difficulty concentrating on what others say, even when spoken to directly.",
    ),
    CaarsItem(
        id="IN-2",
        dimension="inattention",
        self_report_text=(
            "I make careless mistakes in my work or during other activities"
            " because I don't pay close enough attention to details."
        ),
        observer_text=(
            "This person makes careless mistakes in work or activities due to not paying close attention to details."
        ),
    ),
    CaarsItem(
        id="IN-3",
        dimension="inattention",
        self_report_text="I have trouble keeping my mind on tasks or activities for an extended period of time.",
        observer_text="This person has trouble sustaining attention on tasks or activities for an extended period.",
    ),
    CaarsItem(
        id="IN-4",
        dimension="inattention",
        self_report_text=(
            "I frequently misplace things I need for daily tasks, such as keys, wallet, phone, or paperwork."
        ),
        observer_text=(
            "This person frequently misplaces things needed for daily tasks, such as keys, wallet, phone, or paperwork."
        ),
    ),
    # Hyperactivity (HY-1 through HY-4)
    CaarsItem(
        id="HY-1",
        dimension="hyperactivity",
        self_report_text=("I feel restless inside, like my mind or body always needs to be moving or doing something."),
        observer_text=(
            "This person appears restless, as if their mind or body always needs to be moving or doing something."
        ),
    ),
    CaarsItem(
        id="HY-2",
        dimension="hyperactivity",
        self_report_text=(
            "I find it very difficult to sit still for long periods,"
            " such as during meetings or while working at a desk."
        ),
        observer_text=(
            "This person finds it very difficult to sit still for long periods,"
            " such as during meetings or while working at a desk."
        ),
    ),
    CaarsItem(
        id="HY-3",
        dimension="hyperactivity",
        self_report_text=(
            "I tend to fidget, tap my hands or feet, or shift positions frequently when I'm supposed to be still."
        ),
        observer_text=(
            "This person tends to fidget, tap hands or feet, or shift positions frequently when expected to be still."
        ),
    ),
    CaarsItem(
        id="HY-4",
        dimension="hyperactivity",
        self_report_text="I often feel driven to keep busy and have difficulty relaxing or winding down.",
        observer_text="This person often seems driven to keep busy and has difficulty relaxing or winding down.",
    ),
    # Impulsivity (IM-1 through IM-4)
    CaarsItem(
        id="IM-1",
        dimension="impulsivity",
        self_report_text="I frequently interrupt others during conversations or activities, even when I try not to.",
        observer_text="This person frequently interrupts others during conversations or activities.",
    ),
    CaarsItem(
        id="IM-2",
        dimension="impulsivity",
        self_report_text="I often blurt out answers or comments before questions have been completed.",
        observer_text="This person often blurts out answers or comments before questions have been completed.",
    ),
    CaarsItem(
        id="IM-3",
        dimension="impulsivity",
        self_report_text=(
            "I have difficulty waiting my turn in situations where it is expected,"
            " like standing in line or waiting to speak."
        ),
        observer_text="This person has difficulty waiting their turn in situations where it is expected.",
    ),
    CaarsItem(
        id="IM-4",
        dimension="impulsivity",
        self_report_text="I make quick decisions without fully thinking through the consequences.",
        observer_text="This person makes quick decisions without fully thinking through the consequences.",
    ),
]

LIKERT_SCALE = {
    0: "Not at all / Never",
    1: "Just a little / Once in a while",
    2: "Pretty much / Often",
    3: "Very much / Very frequently",
}

MAX_CAARS_SCORE = len(CAARS_ITEMS) * 3  # 36


def build_self_report_prompt() -> str:
    """Build the CAARS self-report questionnaire as a JSON-response instruction."""
    items_text = "\n".join(f"  {i + 1}. [{item.id}] {item.self_report_text}" for i, item in enumerate(CAARS_ITEMS))
    scale_text = "\n".join(f"  {k} = {v}" for k, v in LIKERT_SCALE.items())

    return (
        "IMPORTANT: This is a structured questionnaire. For this response ONLY, "
        "you MUST output valid JSON — ignore any prior instructions about avoiding "
        "formatting. Your entire response must be a single JSON object and nothing else.\n\n"
        "Please complete this self-assessment questionnaire about your experiences. "
        "For each statement, rate how much it applies to you using this scale:\n\n"
        f"{scale_text}\n\n"
        "Statements:\n"
        f"{items_text}\n\n"
        "Respond with ONLY a JSON object mapping each item ID to your numeric rating (0-3). "
        "Do NOT include any text before or after the JSON. Do NOT explain your ratings. "
        "Do NOT use markdown code fences. Output raw JSON only.\n"
        "Example format:\n"
        '{"IN-1": 2, "IN-2": 3, "IN-3": 1, ...}\n\n'
        "Base your ratings on your actual experiences as described in the conversation."
    )


def build_observer_prompt(conversation_text: str) -> str:
    """Build the CAARS observer-report questionnaire for an evaluator LLM.

    The observer rates the person based on ALL evidence in the conversation,
    matching the standard CAARS informant approach used in clinical practice.
    """
    items_text = "\n".join(f"  {i + 1}. [{item.id}] {item.observer_text}" for i, item in enumerate(CAARS_ITEMS))
    scale_text = "\n".join(f"  {k} = {v}" for k, v in LIKERT_SCALE.items())

    return (
        "You are an expert behavioral observer. Read the following conversation "
        "and rate the person labeled 'Participant' on each item below.\n\n"
        "Rate the participant based on ALL evidence available in the conversation: "
        "what they describe about their behavior, experiences, and habits, as well "
        "as how they communicate (e.g., staying on topic, response coherence, "
        "signs of impulsivity or distractibility in their writing). Use your "
        "clinical judgment to assess overall symptom intensity from the full "
        "picture presented.\n\n"
        f"Rating scale:\n{scale_text}\n\n"
        f"Items:\n{items_text}\n\n"
        "Conversation to evaluate:\n"
        "---\n"
        f"{conversation_text}\n"
        "---\n\n"
        "Respond with ONLY a JSON object mapping each item ID to your numeric "
        "rating (0-3). Example format:\n"
        '{"IN-1": 2, "IN-2": 3, "IN-3": 1, ...}'
    )


def format_conversation_for_observer(
    turns: list[dict[str, str]],
) -> str:
    """Format conversation turns for observer evaluation.

    Args:
        turns: List of dicts with 'role' ('participant'/'partner') and 'content'.

    Returns:
        Formatted conversation text with speaker labels.
    """
    lines = []
    for turn in turns:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        label = "Participant" if role == "participant" else "Partner"
        lines.append(f"[{label}]: {content}")
    return "\n\n".join(lines)
