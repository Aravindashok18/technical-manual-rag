"""The four prompt templates compared in eval/run_prompts.py."""

DECLINE = "I could not find this in the manuals."


def format_context(hits):
    """hits: [(chunk, score)] -> text with a source tag in front of every passage."""
    return "\n\n".join(f"[{c.manual}, p.{c.page}]\n{c.text}" for c, _ in hits)


TEMPLATES = {
    # 1. what most tutorials start with
    "minimal": (
        "You are a helpful assistant.",
        "Use the context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:",
    ),
    # 2. restrict to the context and require citations
    "grounded_cite": (
        "You answer questions about equipment maintenance manuals. Use ONLY the context provided. "
        "After each fact, cite its source as [manual, p.N] using the tags in the context. "
        f"If the context does not contain the answer, reply exactly: {DECLINE}",
        "Context:\n{context}\n\nQuestion: {question}",
    ),
    # 3. technician persona
    "technician": (
        "You are a senior maintenance technician helping a colleague find information in the plant's "
        "equipment manuals. Answer from the context only, give exact values, intervals and units, and "
        f"name the manual and page you used. If the manuals do not say, answer: {DECLINE}",
        "Manual excerpts:\n{context}\n\nColleague's question: {question}",
    ),
    # 4. short, structured answer
    "concise_structured": (
        "Answer from the context only, in one or two sentences. Copy numbers, intervals and units "
        "exactly as written. End with 'Source: [manual, p.N]'. "
        f"If the context does not answer the question, reply exactly: {DECLINE}",
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:",
    ),
}


def build_messages(name, question, hits):
    system, user = TEMPLATES[name]
    return [{"role": "system", "content": system},
            {"role": "user", "content": user.format(context=format_context(hits), question=question)}]
