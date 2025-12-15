# import libraries
import os
import re
import json
import tempfile
import warnings
import wave
from typing import List, Tuple, Dict

warnings.filterwarnings("ignore")

import gradio as gr
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from openai import AzureOpenAI
from urllib3.exceptions import NotOpenSSLWarning
import pandas as pd  # for Excel export

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# Config from .env
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "").strip()
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus").strip()
AZURE_SPEECH_VOICE = os.getenv(
    "AZURE_SPEECH_VOICE",
    "en-US-EmmaNeural"
).strip()

# Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

def _speech_config_stt(lang: str = "en-US"):
    cfg = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION
    )
    cfg.speech_recognition_language = lang
    return cfg

def _speech_config_tts() -> speechsdk.SpeechConfig:
    cfg = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION
    )
    cfg.speech_synthesis_voice_name = AZURE_SPEECH_VOICE
    # WAV output format to simplify duration measurement
    cfg.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
    )
    return cfg

# Helper to measure audio duration of WAV files
def get_audio_duration_seconds(path: str) -> float:
    """
    Returns audio duration in seconds for a WAV file.
    Latency is NOT included anywhere: we only sum raw audio duration.
    """
    if not path or not os.path.exists(path):
        return 0.0
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate == 0:
                return 0.0
            return frames / float(rate)
    except Exception as e:
        print(f"Could not read audio duration for {path}: {e}")
        return 0.0

# Fields extracted from the conversation and helper configuration
TABLE_FIELDS = [
    "customer_number",
    "service_type",
    "problem_description",
    "problem_duration",
    "device_type",
    "previous_issue",
    "troubleshooting_done",
    "sentiment_score",
    "intent_category",
    "summary_short",
    "summary_long",
]

# For TECHNICAL calls (intent_category == "Technical")
REQUIRED_FIELDS_TECHNICAL = [
    "customer_number",
    "service_type",
    "problem_description",
    "problem_duration",
    "device_type",
    "previous_issue",
    "troubleshooting_done",
    "sentiment_score",
    "intent_category",
    "summary_short",
    "summary_long",
]

# For GENERAL calls (intent_category in {"Billing","Subscription","Account","Deactivation","Other"})
REQUIRED_FIELDS_GENERAL = [
    "customer_number",
    "intent_category",
    "sentiment_score",
    "summary_short",
    "summary_long",
]

SERVICE_TYPES = ["Television", "Broadband", "Mobile", "Landline", "Other"]
SENTIMENT_LEVELS = ["positive", "neutral", "negative"]

GENERAL_INTENTS = ["billing", "subscription", "account", "deactivation", "other"]
INTENT_CATEGORIES = ["Technical", "Billing", "Subscription", "Account", "Deactivation", "Other"]

def _get_required_fields(collected: Dict[str, str]) -> List[str]:
    """
    Two-level logic:
    - If intent_category == 'Technical'  -> technical schema.
    - Else -> general schema (only minimal info required).
    - If sentiment is negative, reduce required fields to essentials only.
    """
    sentiment = (collected.get("sentiment_score") or "").lower().strip()
    negative = sentiment in ["negative"]

    intent = (collected.get("intent_category") or "").strip().lower()

    if intent == "technical":
        base = [
            "customer_number",
            "service_type",
            "problem_description",
            "problem_duration",
            "device_type",
            "previous_issue",
            "troubleshooting_done",
            "sentiment_score",
            "intent_category",
            "summary_short",
            "summary_long",
        ]
        if negative:
            return [
                "customer_number",
                "service_type",
                "problem_description",
                "sentiment_score",
                "intent_category",
                "summary_short",
                "summary_long",
            ]
        return base

    if intent in GENERAL_INTENTS:
        base = [
            "customer_number",
            "intent_category",
            "sentiment_score",
            "summary_short",
            "summary_long",
        ]
        if negative:
            return [
                "customer_number",
                "intent_category",
                "sentiment_score",
                "summary_short",
                "summary_long",
            ]
        return base

    # If intent is not yet clear, fall back to the general schema to ensure basic triage.
    return REQUIRED_FIELDS_GENERAL

# Speech-to-text (STT) and text-to-speech (TTS) helpers
def transcribe_audio_speech(audio_path: str, language: str = "en-US") -> str:
    if not audio_path or not os.path.exists(audio_path):
        return ""
    try:
        audio_cfg = speechsdk.AudioConfig(filename=audio_path)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=_speech_config_stt(language),
            audio_config=audio_cfg
        )
        res = recognizer.recognize_once_async().get()
        if res.reason == speechsdk.ResultReason.RecognizedSpeech:
            return res.text or ""
        if res.reason == speechsdk.ResultReason.Canceled and res.cancellation_details:
            print("STT canceled:", res.cancellation_details.reason, res.cancellation_details.error_details)
        return ""
    except Exception as e:
        print(f"STT error: {e}")
        return ""

def _synthesize_to_wav(text: str, out_path: str):
    """
    Low-level helper: sends text to Azure TTS and writes to out_path (WAV).
    """
    cfg = _speech_config_tts()
    audio_out = speechsdk.audio.AudioOutputConfig(filename=out_path)
    synth = speechsdk.SpeechSynthesizer(
        speech_config=cfg,
        audio_config=audio_out
    )

    # Single call; text is short enough for this use case
    result = synth.speak_text_async(text).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        err = ""
        if result.reason == speechsdk.ResultReason.Canceled and result.cancellation_details:
            err = result.cancellation_details.error_details or str(result.cancellation_details.reason)
        raise RuntimeError(f"TTS synthesis failed: {err}")

def speak_text_to_file(
    text: str,
    metrics: Dict[str, float] = None,
) -> str:
    """
    Synthesizes text to a WAV file and updates:
    - ai_talk_time_sec: sum of AI audio durations (no latency)
    Adds a small robustness check: if the file is empty, retry once.
    """
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    # First attempt
    try:
        _synthesize_to_wav(text, out_path)
    except Exception as e:
        print("TTS error on first attempt:", e)
        return None

    # Check file size; if 0, retry once with a new file
    try:
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            print("TTS produced empty file; retrying once.")
            out_path_retry = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            _synthesize_to_wav(text, out_path_retry)
            out_path = out_path_retry
    except Exception as e:
        print("Error checking TTS file size:", e)

    # Measure AI talk time (audio duration only, not latency)
    if metrics is not None and out_path and os.path.exists(out_path):
        dur = get_audio_duration_seconds(out_path)
        metrics["ai_talk_time_sec"] = metrics.get("ai_talk_time_sec", 0.0) + dur

    return out_path

# Core conversation logic and system prompts
def all_collected(collected: Dict[str, str]) -> bool:
    """
    Decide if the agent has collected everything relevant based on call type:
    - Technical: require technical details.
    - General: require only minimal generic info.
    """
    required = _get_required_fields(collected)
    return all((collected.get(k) or "").strip() for k in required)

def build_agent_system_prompt(collected: Dict[str, str]) -> str:
    """
    System prompt for the NOS virtual assistant with improved realism:
    - Urgency-aware question reduction
    - Automatic inference of fields when possible
    - No repeated or unnecessary questions
    - Human-like tone, friendly and concise
    - More natural troubleshooting phrasing
    - Apologies only when appropriate (not in every message)
    """

    required = _get_required_fields(collected)
    missing = [k for k in required if not (collected.get(k) or "").strip()]
    missing_text = ", ".join(missing) if missing else "none"

    return f"""You are a NOS customer service virtual assistant. Speak English only.

Tone & Style:
- Warm, natural, friendly, and human-like.
- Keep replies concise (1–3 sentences).
- Use a customer-service tone that is polite, calm, and professional.
- Only apologize when the issue indicates a clear NOS-side problem (service degradation, poor connection, outages, technical failures, dissatisfaction with service performance).
- Do NOT apologize when the customer is making a request unrelated to a service failure (e.g., billing, payment date changes, subscription changes, upgrades). Instead, use helpful, supportive language such as:
  "Let me see what I can do for you."
  "I can help with that."
- Do not repeat long portions of what the customer said; acknowledge briefly.

Task:
You gather a small set of information so a human specialist can assist the customer efficiently.
Once all relevant information is collected, do NOT ask any more questions.
Instead, thank the customer naturally and hand over.

STRICT RULE:
- For TECHNICAL calls, the assistant may NOT conclude the conversation until ALL required technical fields are filled, unless:
    (a) the customer expresses clear frustration or stress, AND
    (b) sentiment_score = "negative".
- In all other cases, the assistant must continue asking one missing field at a time.

Important: 
- For both technical and general inquiries, the assistant should make a clear effort to collect ALL required fields as long as the customer's sentiment is not strongly negative.
- If sentiment_score is "negative" AND the customer expresses frustration, urgency, or stress explicitly, the assistant may skip non-essential questions and move directly toward handover.
- Otherwise, the assistant MUST attempt to collect every required field according to call type.

### Absolutely avoid:
- Asking for information the customer already provided (directly or indirectly).
- Repeating long parts of what the customer said.
- Overly robotic questions (e.g., "Describe the problem in detail").
- Asking for "problem description" again if the issue is already clear (e.g., Wi-Fi not working).
- Using generic empathy phrases like "I understand how frustrating that can be" without also giving a clear, concrete apology.
- Never add extra empathy sentences after the apology. No phrases like:
  "I understand how frustrating that can be."
  "I know this must be annoying."
  "I get how stressful this is."
One apology sentence only.
- Avoid apologizing for issues that are not NOS's fault, such as billing preferences, payment date changes, plan changes, upgrades, or general administrative requests.
- Avoid mixing apology + sales language.
- Never claim "I have all the information I need" if technical fields are missing and sentiment is not negative.

### Inference-first rules:
Always infer fields whenever reasonably possible:
- For TECHNICAL calls: If the customer mentions "Wi-Fi not working" or "internet down" → service_type = Broadband.
- For TECHNICAL calls: If they mention a router, restart attempts, Wi-Fi → device_type = Router.
- If they complain about billing, charges, invoices → intent_category = Billing.
- If they mention cancelling or activating → intent_category = Subscription or Deactivation.
- If core problem is already described → problem_description should be inferred, do NOT ask again.
- For GENERAL calls (Billing, Subscription, Account, Deactivation, Other): Do NOT ask for service_type. Only infer it if the customer explicitly mentions it (e.g., "my mobile plan", "my TV service"). Never ask for it.
Only ask if the meaning is genuinely unclear.

Apology decision:
- If intent_category = Technical → Apology required.
- If the customer expresses disappointment with service quality → Apology required.
- If intent_category in ["Billing", "Subscription", "Account", "Deactivation", "Other"] → No apology.

### Urgency-aware question reduction:
If customer sounds frustrated, urgent, or stressed:
- Ask ONLY the essential missing fields.
- Skip minor details unless required for routing.
- Aim to conclude in as few turns as possible.

### Stricter collection rule:
If sentiment is neutral or positive, you MUST collect every required field for the call type.
Only relax this requirement if the customer expresses clear frustration, urgency, or emotional distress.

### For technical issues (STRICT COLLECTION MODE):
For all TECHNICAL calls:
- The assistant MUST collect ALL technical fields:
  customer_number,
  service_type,
  problem_description,
  problem_duration,
  device_type,
  previous_issue,
  troubleshooting_done,
  sentiment_score,
  intent_category.
- This is mandatory unless the customer shows clear frustration or negative sentiment.
- If sentiment_score is NOT "negative":
    → Keep asking missing fields one by one until all are collected.
- If sentiment_score IS "negative":
    → Skip further questions and immediately hand over to a human.

### For GENERAL calls:
For GENERAL calls (Billing, Subscription, Account, Deactivation, Other):
- Required fields must still be collected (customer_number, service_type if explicitly stated, sentiment_score, intent_category).
- As long as sentiment is not strongly negative, the assistant MUST explicitly ask for customer_number if missing.

### Hand-over logic:
For TECHNICAL calls:
- Only hand over when ALL technical fields are filled.
- Do NOT hand over early unless sentiment is negative.

For GENERAL calls:
- Hand over when general-required fields are filled.

When handing over:
- Do NOT ask further questions.
- Your last message must NEVER include a question.
- Hand over with a natural sentence such as:
  "Thanks, I have all the information I need. I'll connect you with a specialist now."

### Required fields (based on call type):
Missing: {missing_text}

Unless the customer's sentiment is strongly negative, you MUST ask questions to fill the missing fields.
If sentiment is strongly negative, skip only non-essential questions.

Your job:
- When the customer reports a technical problem or dissatisfaction with service quality, begin with one short apology sentence.
- When the customer requests something non-technical (billing, subscription, account updates, upgrades), do NOT apologize. Instead respond with a helpful, service-oriented introduction such as:
  "I can help with that. Could you please share your customer number?"
- Ask exactly ONE question only if something essential is still missing.
- If everything is collected, give a natural handover message.

Remember:
- Stay empathetic.
- Keep it short.
- Sound like a real NOS agent, not a checklist bot.

Apology style rule:
If generating an apology, it must be one sentence only, containing only the apology itself. 
Example format:
"I'm sorry you're having trouble with your internet."
Do not add:
- Explanations
- Empathy commentary
- Emotional reflections
- "I understand…" or similar
"""

def extract_fields_llm(
    chat_history: List[Tuple[str, str]],
    current: Dict[str, str],
    metrics: Dict[str, float]
):
    conversation = "\n".join([f"{s}: {m}" for s, m in chat_history])

    extraction_prompt = f"""
You are an information extractor. Return ONLY a single valid JSON object with the following keys in this exact order:
{json.dumps(TABLE_FIELDS)}

Field rules:
- customer_number: string of digits only (remove spaces and non-digits). If unknown, "".
- service_type: Only fill this for TECHNICAL calls or if the customer explicitly stated it (e.g., "my mobile plan", "my TV service"). For GENERAL calls (Billing, Subscription, Account, Deactivation, Other), if not explicitly mentioned, leave "" and do NOT try to infer it. If unknown, "".
- problem_description: concise 1–2 sentences. If unknown, "".
- problem_duration: normalize to something like "2 hours", "3 days", or "since yesterday". If unknown, "".
- device_type: short free text like "router", "SIM card", "TV box". If unknown, "".
- previous_issue: "0" or "1". If unknown, "".
- troubleshooting_done: "0" or "1". If unknown, "".
- sentiment_score: one of {json.dumps(SENTIMENT_LEVELS)} inferred from tone; if unknown, "".
- intent_category: EXACTLY one of {json.dumps(INTENT_CATEGORIES)}. If unknown, "".
- summary_short: 1 sentence summary for quick GC reference. If unknown, "".
- summary_long: 2–3 sentence concise overview including context and actions tried or requested. If unknown, "".

Special rules:
- If intent_category == "Technical":
  - Try to infer and fill service_type, problem_description, problem_duration, device_type,
    previous_issue, and troubleshooting_done whenever possible.
- If intent_category is NOT "Technical" (Billing, Subscription, Account, Deactivation, Other):
  - It is OK if service_type, problem_description, problem_duration, device_type,
    previous_issue, and troubleshooting_done remain "".
  - For service_type: Only fill if the customer explicitly mentioned it. Do NOT infer it.
  - Still provide a good summary_short and summary_long.

Merging rules:
- Start from these previous values and KEEP them unless the user corrected them:
{json.dumps(current, ensure_ascii=False, indent=2)}
- Only overwrite a field if new info is present or a correction was made.
- If a value is still unknown, leave it as "".

Conversation:
{conversation}
"""

    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "system", "content": extraction_prompt}],
        temperature=0.0,
        max_tokens=500
    )

    # Count tokens for extraction call
    if metrics is not None and getattr(resp, "usage", None) is not None:
        u = resp.usage
        metrics["prompt_tokens_total"] = metrics.get("prompt_tokens_total", 0) + (u.prompt_tokens or 0)
        metrics["completion_tokens_total"] = metrics.get("completion_tokens_total", 0) + (u.completion_tokens or 0)

    text = (resp.choices[0].message.content or "").strip()

    try:
        m = re.search(r"\{.*\}", text, flags=re.S)
        parsed = json.loads(m.group(0)) if m else {}
    except Exception:
        parsed = {}

    merged = {k: (parsed.get(k) or current.get(k) or "").strip() for k in TABLE_FIELDS}

    # Normalize customer_number: digits only
    merged["customer_number"] = re.sub(r"\D+", "", merged.get("customer_number", ""))

    # Normalize service_type
    if merged.get("service_type") and merged["service_type"] not in SERVICE_TYPES:
        st = merged["service_type"].lower()
        if "tv" in st:
            merged["service_type"] = "Television"
        elif "broad" in st or "internet" in st or "fiber" in st:
            merged["service_type"] = "Broadband"
        elif "mobile" in st or "cell" in st or "sim" in st:
            merged["service_type"] = "Mobile"
        elif "land" in st or "fixed" in st or "phone" in st:
            merged["service_type"] = "Landline"
        else:
            merged["service_type"] = "Other"

    # Normalize sentiment_score
    if merged.get("sentiment_score") and merged["sentiment_score"] not in SENTIMENT_LEVELS:
        merged["sentiment_score"] = merged["sentiment_score"].lower()

    # Normalize binary fields
    for b in ["previous_issue", "troubleshooting_done"]:
        v = (merged.get(b, "") or "").strip().lower()
        if v in ["yes", "y", "true", "1"]:
            merged[b] = "1"
        elif v in ["no", "n", "false", "0"]:
            merged[b] = "0"
        elif v not in ["", "0", "1"]:
            merged[b] = ""

    return merged

def query_agent(messages, metrics: Dict[str, float]):
    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=messages,
        temperature=0.5,
        max_tokens=300
    )

    if metrics is not None and getattr(resp, "usage", None) is not None:
        u = resp.usage
        metrics["prompt_tokens_total"] = metrics.get("prompt_tokens_total", 0) + (u.prompt_tokens or 0)
        metrics["completion_tokens_total"] = metrics.get("completion_tokens_total", 0) + (u.completion_tokens or 0)

    return (resp.choices[0].message.content or "").strip()

# Build the HTML chat transcript shown in the UI
def format_chat(history: List[Tuple[str, str]]) -> str:
    html = ""
    for speaker, msg in history:
        is_agent = speaker.lower() in ["agent", "nos"]
        avatar_bg = "#7CB800" if is_agent else "#9aa1a7"
        avatar_text = "A" if is_agent else "C"
        html += f"""
        <div class="chat-bubble">
            <div class="chat-avatar" style="background-color: {avatar_bg};">{avatar_text}</div>
            <div class="chat-message">{msg}</div>
        </div>
        """
    return html

# Gradio handlers for starting calls and processing audio
def start_or_export(started: bool, chat_history, collected, metrics):
    if not started:
        # Start call
        greeting_text = "Welcome to NOS customer service. You're speaking with a virtual assistant. I'll gather some information so one of our specialists can help you as quickly as possible. How can I assist today?"
        history = [("Agent", greeting_text)]

        collected = {k: "" for k in TABLE_FIELDS}
        metrics = {
            "ai_talk_time_sec": 0.0,
            "customer_talk_time_sec": 0.0,
            "stt_chars": 0,
            "tts_chars": 0,
            "prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "turns": 0,
            "negative_turns": 0,
            "stt_fail_streak": 0,
            "intent_history": [],
        }

        metrics["tts_chars"] += len(greeting_text)
        audio_path = speak_text_to_file(greeting_text, metrics=metrics)

        html = format_chat(history)
        player_html = audio_path

        return (
            html,
            history,
            gr.update(interactive=True, value=None, visible=True),
            player_html,
            collected,
            gr.update(value="Export Summary"),
            gr.update(visible=False, value=None),  # JSON file hidden
            gr.update(visible=False, value=None),  # Excel file hidden
            True,
            metrics,
        )
    else:
        # Export both JSON for GC + Excel metrics

        # 1) JSON for GC: only TABLE_FIELDS + full_transcript
        json_data = {k: collected.get(k, "") for k in TABLE_FIELDS}

        # Add full transcript
        if chat_history:
            transcript_lines = []
            for speaker, message in chat_history:
                transcript_lines.append(f"{speaker}: {message}")
            json_data["full_transcript"] = "\n".join(transcript_lines)
        else:
            json_data["full_transcript"] = ""

        json_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        # 2) Excel metrics row
        intent = (collected.get("intent_category") or "").strip().lower()
        category = "technical" if intent == "technical" else "general"

        # Content fields for fill-percentage (summary_short/summary_long excluded)
        if category == "technical":
            content_keys = [
                "customer_number",
                "service_type",
                "problem_description",
                "problem_duration",
                "device_type",
                "previous_issue",
                "troubleshooting_done",
                "sentiment_score",
                "intent_category",
            ]
            denom = 9
        else:
            content_keys = [
                "customer_number",
                "service_type",
                "intent_category",
                "sentiment_score",
            ]
            denom = 4

        filled_count = sum(
            1 for k in content_keys if (collected.get(k) or "").strip()
        )
        pct_filled_fields = filled_count / denom if denom > 0 else 0.0

        metrics_row = {
            "category": category,
            "ai_talk_time_sec": metrics.get("ai_talk_time_sec", 0.0) if metrics else 0.0,
            "customer_talk_time_sec": metrics.get("customer_talk_time_sec", 0.0) if metrics else 0.0,
            "stt_chars": metrics.get("stt_chars", 0) if metrics else 0,
            "tts_chars": metrics.get("tts_chars", 0) if metrics else 0,
            "prompt_tokens_total": metrics.get("prompt_tokens_total", 0) if metrics else 0,
            "completion_tokens_total": metrics.get("completion_tokens_total", 0) if metrics else 0,
            "turns": metrics.get("turns", 0) if metrics else 0,
            "pct_filled_fields": pct_filled_fields,
        }

        df = pd.DataFrame([metrics_row], columns=[
            "category",
            "ai_talk_time_sec",
            "customer_talk_time_sec",
            "stt_chars",
            "tts_chars",
            "prompt_tokens_total",
            "completion_tokens_total",
            "turns",
            "pct_filled_fields",
        ])
        excel_path = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx").name
        df.to_excel(excel_path, index=False)

        return (
            gr.update(),  # chatbot
            gr.update(),  # state
            gr.update(),  # audio_input
            gr.update(),  # player_panel
            gr.update(),  # collected_state
            gr.update(value="Start Call"),  # button label reset if you want
            gr.update(value=json_path, visible=True),   # JSON summary
            gr.update(value=excel_path, visible=True),  # Excel metrics
            True,
            metrics,
        )

def respond_voice(audio_file, chat_history, collected, metrics):
    if chat_history is None:
        chat_history = []
    if collected is None:
        collected = {k: "" for k in TABLE_FIELDS}
    if metrics is None:
        metrics = {
            "ai_talk_time_sec": 0.0,
            "customer_talk_time_sec": 0.0,
            "stt_chars": 0,
            "tts_chars": 0,
            "prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "turns": 0,
            "negative_turns": 0,
            "stt_fail_streak": 0,
            "intent_history": [],
        }

    # Early return if no audio file provided
    if not audio_file or not os.path.exists(audio_file):
        return (
            format_chat(chat_history),
            chat_history,
            gr.update(value=None, interactive=True),
            None,
            collected,
            metrics,
        )

    # Customer talk time: audio duration only (no latency)
    dur_cust = get_audio_duration_seconds(audio_file)
    metrics["customer_talk_time_sec"] = metrics.get("customer_talk_time_sec", 0.0) + dur_cust

    transcript = transcribe_audio_speech(audio_file, language="en-US")
    if transcript:
        metrics["stt_chars"] = metrics.get("stt_chars", 0) + len(transcript)
        metrics["stt_fail_streak"] = 0
    else:
        # No recognized speech: implement STT-failure handover rule
        fail = metrics.get("stt_fail_streak", 0) + 1
        metrics["stt_fail_streak"] = fail

        if fail == 1:
            agent_reply = "I couldn't hear you clearly. Could you please repeat that?"
        else:
            agent_reply = "I'm having trouble understanding the audio. I'll connect you with a specialist now."

        metrics["tts_chars"] = metrics.get("tts_chars", 0) + len(agent_reply)

        last_speaker = chat_history[-1][0] if chat_history else None
        if last_speaker is not None and last_speaker != "Agent":
            metrics["turns"] = metrics.get("turns", 0) + 1

        chat_history.append(("Agent", agent_reply))
        audio_path = speak_text_to_file(agent_reply, metrics=metrics)
        player_html = audio_path if audio_path and os.path.exists(audio_path) else None

        return (
            format_chat(chat_history),
            chat_history,
            gr.update(value=None, interactive=True),
            player_html,
            collected,
            metrics,
        )

    if not transcript.strip():
        # Should be covered above, but keep safe fallback
        return (
            format_chat(chat_history),
            chat_history,
            gr.update(value=None, interactive=True),
            None,
            collected,
            metrics,
        )

    # Track turns: last speaker -> now customer
    last_speaker = chat_history[-1][0] if chat_history else None
    if last_speaker is not None and last_speaker != "Customer":
        metrics["turns"] = metrics.get("turns", 0) + 1

    chat_history.append(("Customer", transcript))

    # Extract and merge structured info
    try:
        collected = extract_fields_llm(chat_history, collected, metrics)
    except Exception as e:
        print("Extraction error:", e)

    # Sentiment & intent for rule logic
    sentiment = (collected.get("sentiment_score") or "").lower().strip()
    intent = (collected.get("intent_category") or "").lower().strip()
    is_technical = intent == "technical"

    # Track negative sentiment streak
    if sentiment == "negative":
        metrics["negative_turns"] = metrics.get("negative_turns", 0) + 1
    else:
        metrics["negative_turns"] = 0

    # Track intent history for "uncertain intent"
    history = metrics.get("intent_history", [])
    if intent:
        if not history or history[-1] != intent:
            history.append(intent)
    metrics["intent_history"] = history
    distinct_intents = {i for i in history if i}

    turns = metrics.get("turns", 0)
    all_required_filled = all_collected(collected)

    # Handover decision logic
    handover_reason = None

    # 1) Negative sentiment across multiple turns
    if metrics.get("negative_turns", 0) >= 2:
        handover_reason = "negative_sentiment"
    # 2) Turn limit exceeded
    elif turns > 10:
        handover_reason = "max_turns"
    # 3) Uncertain or conflicting intent
    elif (not intent and turns >= 6) or (len(distinct_intents) >= 2 and not all_required_filled):
        handover_reason = "uncertain_intent"
    # 4) All required info collected
    elif all_required_filled:
        handover_reason = "all_info_collected"

    if handover_reason is not None:
        if handover_reason == "all_info_collected":
            agent_reply = "Thanks, I have all the information I need. I'll connect you with a specialist now."
        elif handover_reason == "max_turns":
            agent_reply = "I'll connect you with a specialist now so they can help you directly."
        elif handover_reason == "negative_sentiment":
            agent_reply = "I'll connect you with a specialist now so they can assist you further."
        else:  # uncertain_intent
            agent_reply = "I'll connect you with a specialist now so they can better understand and resolve your issue."

        metrics["tts_chars"] = metrics.get("tts_chars", 0) + len(agent_reply)

        last_speaker = chat_history[-1][0] if chat_history else None
        if last_speaker is not None and last_speaker != "Agent":
            metrics["turns"] = metrics.get("turns", 0) + 1

        chat_history.append(("Agent", agent_reply))
        audio_path = speak_text_to_file(agent_reply, metrics=metrics)
        player_html = audio_path if audio_path and os.path.exists(audio_path) else None

        return (
            format_chat(chat_history),
            chat_history,
            gr.update(value=None, interactive=True),
            player_html,
            collected,
            metrics,
        )

    # If no hard handover rule triggered, go through LLM agent
    system_prompt = build_agent_system_prompt(collected)
    messages = [{"role": "system", "content": system_prompt}]
    for speaker, msg in chat_history:
        role = "user" if speaker == "Customer" else "assistant"
        messages.append({"role": role, "content": msg})

    try:
        agent_reply = query_agent(messages, metrics)
    except Exception as e:
        agent_reply = f"Sorry, something went wrong: {e}"
        if all_required_filled:
            agent_reply = "Thank you. I will now forward you to a human agent who will help solve the issue."

    if agent_reply:
        metrics["tts_chars"] = metrics.get("tts_chars", 0) + len(agent_reply)

    # Track turns for agent reply
    last_speaker = chat_history[-1][0] if chat_history else None
    if last_speaker is not None and last_speaker != "Agent":
        metrics["turns"] = metrics.get("turns", 0) + 1

    chat_history.append(("Agent", agent_reply))
    audio_path = speak_text_to_file(agent_reply, metrics=metrics)
    player_html = audio_path if audio_path and os.path.exists(audio_path) else None

    # Mic / record button stays interactive in all cases
    return (
        format_chat(chat_history),
        chat_history,
        gr.update(value=None, interactive=True),
        player_html,
        collected,
        metrics,
    )

# Gradio Blocks layout for the web app
GREEN = "#7CB800"

with gr.Blocks(
    theme=gr.themes.Base(),
    css=f"""
    body {{ font-family: 'Inter', sans-serif; margin: 0; }}
    #header {{ position: fixed; top: 0; left: 0; right: 0; background: white; z-index: 1000; padding: 12px 24px 0 24px; }}
    #header-logo img {{ border: none !important; border-bottom: 4px solid {GREEN}; width: 100%; height: auto; max-height: 100px; object-fit: contain; }}
    #main-container {{ display: flex; flex-direction: column; margin-top: 120px; height: calc(100vh - 120px); }}
    #chat-wrapper {{ flex: 1; overflow-y: auto; max-height: calc(100vh - 260px); padding: 0 24px 220px 24px; scroll-behavior: smooth; }}
    #chat-container {{ padding: 20px 0 0 0; }}
    .chat-bubble {{ display: flex; align-items: flex-start; margin-bottom: 16px; gap: 12px; }}
    .chat-avatar {{ width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; }}
    .chat-message {{ background-color: #e6ecf0; padding: 12px 16px; border-radius: 12px; max-width: 85%; position: relative; }}
    .chat-message::before {{ content: ""; position: absolute; top: 10px; left: -10px; width: 0; height: 0; border-top: 8px solid transparent; border-bottom: 8px solid transparent; border-right: 10px solid #e6ecf0; }}
    .primary-btn {{ background: {GREEN}; color: #fff; font-weight: 700; border-radius: 10px; padding: 12px 18px; width: 100%; }}
    .footer-bar {{ position: fixed; bottom: 0; left: 0; right: 0; padding: 12px 24px; background: white; z-index: 120; display: flex; align-items: center; gap: 16px; }}
    .mic-bar {{ position: fixed; bottom: 72px; left: 24px; right: 24px; z-index: 110; background: white; padding: 10px 12px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
    """
) as demo:

    # Header
    with gr.Row(elem_id="header"):
        with gr.Column(scale=1):
            gr.Image(
                value="header_2.png",
                show_label=False,
                show_download_button=False,
                show_share_button=False,
                container=False,
                elem_id="header-logo"
            )

    # Chat area
    with gr.Column(elem_id="main-container"):
        chat_wrapper = gr.Column(visible=True, elem_id="chat-wrapper")
        with chat_wrapper:
            with gr.Column(elem_id="chat-container"):
                chatbot = gr.HTML(value="<div id='chatbox'></div>")

    # Fixed mic bar
    mic_bar = gr.Column(visible=True, elem_classes=["mic-bar"])
    with mic_bar:
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Press Record, speak, then Stop",
            interactive=True,
            show_label=True
        )

    # Agent audio output; update with file path on each reply and autoplay
    player_panel = gr.Audio(
        visible=True,
        interactive=False,
        autoplay=True,
        show_label=False
    )

    # State
    state = gr.State([])  # chat history
    collected_state = gr.State({k: "" for k in TABLE_FIELDS})
    metrics_state = gr.State({
        "ai_talk_time_sec": 0.0,
        "customer_talk_time_sec": 0.0,
        "stt_chars": 0,
        "tts_chars": 0,
        "prompt_tokens_total": 0,
        "completion_tokens_total": 0,
        "turns": 0,
        "negative_turns": 0,
        "stt_fail_streak": 0,
        "intent_history": [],
    })
    started_state = gr.State(False)

    # Footer: Start first, then Export
    with gr.Row(elem_classes=["footer-bar"]):
        start_export_btn = gr.Button("Start Call", elem_classes=["primary-btn"])
        summary_json_file = gr.File(label="Download JSON (GC)", visible=False)
        metrics_excel_file = gr.File(label="Download metrics (Excel)", visible=False)

    # Start/Export button
    start_export_btn.click(
        fn=start_or_export,
        inputs=[started_state, state, collected_state, metrics_state],
        outputs=[
            chatbot,
            state,
            audio_input,
            player_panel,
            collected_state,
            start_export_btn,
            summary_json_file,
            metrics_excel_file,
            started_state,
            metrics_state,
        ],
    )

    # Audio-only conversation
    audio_input.change(
        fn=respond_voice,
        inputs=[audio_input, state, collected_state, metrics_state],
        outputs=[
            chatbot,
            state,
            audio_input,
            player_panel,
            collected_state,
            metrics_state,
        ],
        show_progress=False,
        queue=False,  # avoid queued race conditions
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False, show_api=False)
