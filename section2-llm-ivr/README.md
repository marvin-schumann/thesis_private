# Section 2: LLM-Powered IVR Prototype

**Author:** Runa Maria Kleppek

## Research Question

Can a Large Language Model (LLM) improve the Interactive Voice Response (IVR) experience by conducting natural conversations and generating structured handover packages for call center agents?

## Methodology

- **Interface**: Gradio web application for voice interaction
- **LLM**: GPT-4o via OpenAI API
- **Speech-to-Text**: OpenAI Whisper
- **Text-to-Speech**: OpenAI TTS
- **Conversation Management**: Multi-turn dialogue with context retention

## Key Features

The prototype demonstrates an LLM-powered IVR system that:
1. Conducts natural language conversations with customers
2. Identifies customer intent and service requirements
3. Extracts relevant information through conversational flow
4. Generates structured handover packages for agents

## Output

The system produces a structured handover package containing:
- Customer number
- Service type
- Problem description
- Customer sentiment
- Recommended next steps

## Dependencies

This section is **standalone** and does not depend on other sections.

## Files

| File | Description |
|------|-------------|
| `nos_voice_ivr_prototype.py` | Main Gradio application with LLM integration |
| `env.env` | Environment variables (API keys) |

## How to Run

1. Configure API keys in `env.env`
2. Install dependencies: `pip install gradio openai`
3. Run the application:

```bash
cd section2-llm-ivr
python nos_voice_ivr_prototype.py
```

## Requirements

- Python 3.8+
- OpenAI API key (for GPT-4o, Whisper, and TTS)
- Gradio

## Notes

- API keys in `env.env` should be kept confidential
- The prototype is designed for demonstration purposes
- Voice interaction requires microphone access in the browser
