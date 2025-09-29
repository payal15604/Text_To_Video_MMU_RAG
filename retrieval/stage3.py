from transformers import pipeline
from typing import List, Dict
import torch
# -------------------------------
# 1. Create Context Document
# -------------------------------
def create_context_document(live_context, archival_context_chunks):
    """
    Build a structured context document combining live and archival data.
    """
    doc = "[START OF LIVE WEB CONTEXT]\n"
    doc += live_context.strip() + "\n"
    doc += "[END OF LIVE WEB CONTEXT]\n\n"

    doc += "[START OF ARCHIVAL CONTEXT]\n"
    for idx, chunk in enumerate(archival_context_chunks, 1):
        doc += f"{chunk.strip()}\n"
        if idx != len(archival_context_chunks):
            doc += "---\n"
    doc += "[END OF ARCHIVAL CONTEXT]\n"

    return doc

def build_context_doc(live_context, archival_context):
    """Builds unified context doc from live + archival data."""
    doc = f"""
    [START OF LIVE WEB CONTEXT]
    {live_context}
    [END OF LIVE WEB CONTEXT]

    [START OF ARCHIVAL CONTEXT]
    {"---\n".join(archival_context)}
    [END OF ARCHIVAL CONTEXT]
    """
    print(len(doc))
    return doc.strip()

#"mistralai/Mistral-7B-Instruct-v0.2"
# -------------------------------
# 2. Load Mistral-7B-Instruct (HF)
# -------------------------------
def load_summarizer(model_name="microsoft/phi-3-mini-4k-instruct", device="cuda"):
    """
    Load a Hugging Face pipeline for text generation.
    """
    summarizer = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto"
    )
    return summarizer

# ---------------------------
# Synthesis Function
# ---------------------------
def synthesize_summary(summarizer, context_doc, max_new_tokens=300):
    """Generate synthesized summary using Mistral."""
    prompt = f"""
    You are an expert factual synthesizer. Merge, summarize,
    and de-duplicate the following information into a structured,
    thematic summary. Use clear headings and short paragraphs.
    Do not invent facts.

    {context_doc}
    """
    response = summarizer(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


    return response[0]["generated_text"]

# ---------------------------
# Main Stage 3 Entrypoint
# ---------------------------
def stage3(live_context, archival_context):
    """Full Stage 3 pipeline."""
    print('Loading summarizer')
    summarizer = load_summarizer()
    print('building context doc')
    context_doc = build_context_doc(live_context, archival_context)
    print('synthesizing summary')
    summary = synthesize_summary(summarizer, context_doc)
    return summary