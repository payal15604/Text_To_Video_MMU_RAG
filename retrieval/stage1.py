
import nest_asyncio
import asyncio
import json
import re
from transformers import pipeline
from playwright.async_api import async_playwright

nest_asyncio.apply()

# ------------------------------
# Part A: Live Web Scraper
# ------------------------------
async def scrape_news(query, num_results=3):
    """Scrape top news results for a given query using Playwright (Google News)."""
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Go to Google News
        await page.goto(f"https://news.google.com/search?q={query}")

        # Grab article titles + snippets
        articles = await page.query_selector_all("article")
        for art in articles[:num_results]:
            title_el = await art.query_selector("h3")
            snippet_el = await art.query_selector("span")
            title = await title_el.inner_text() if title_el else "No title"
            snippet = await snippet_el.inner_text() if snippet_el else ""
            results.append(f"{title}: {snippet}")

        await browser.close()
    return results

# ------------------------------
# Live Bing Search Scraper
# ------------------------------

async def scrape_bing(query, num_results=5):
    """Scrape top search results from Bing using Playwright."""
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(f"https://www.bing.com/search?q={query}")

        # Bing result blocks
        items = await page.query_selector_all("li.b_algo")
        for item in items[:num_results]:
            title_el = await item.query_selector("h2 a")
            snippet_el = await item.query_selector("p")

            title = await title_el.inner_text() if title_el else "No title"
            snippet = await snippet_el.inner_text() if snippet_el else "No snippet"

            results.append({"title": title, "snippet": snippet})

        await browser.close()
    return results

# ------------------------------
# Part B: Query Expansion with LLM  
# ------------------------------

query_expander = pipeline(
    "text-generation",
    model="microsoft/phi-3-mini-4k-instruct",
    device_map="auto"
)


def expand_query(user_query: str):
    """Expand query into its most relevant interpretation + sub-questions using Phi-3-mini."""
    prompt = f"""
You are a research assistant.

Take the following user query and do exactly two things:
1. Keep the query as-is (do not reinterpret or create alternate versions).
2. Break it down into 3–5 factual sub-questions.

Output MUST be a single JSON object in this format:

{{
  "query": "<original user query>",
  "sub_questions": [
    "...",
    "...",
    "..."
  ]
}}

User Query: "{user_query}"
"""
    response = query_expander(prompt, max_new_tokens=300, do_sample=False)
    raw_text = response[0]["generated_text"]

    # Strip off the prompt so we only keep the model's answer
    cleaned = raw_text.replace(prompt, "").strip()

    # Extract first JSON object if multiple are present
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    # Parse JSON safely
    try:
        data = json.loads(cleaned)
        return data  # { "query": "...", "sub_questions": ["...", "..."] }
    except Exception:
        return {
            "query": user_query,
            "sub_questions": [cleaned]  # fallback if parse fails
        }

# -------------------------
# Stage 1 Orchestration
# -------------------------
async def stage1(user_query: str):
    print(f"🔍 Running Stage 1 for query: {user_query}\n")

    # Run both in parallel
    #live_news = await scrape_news(user_query, num_results=3)
    sub_queries = expand_query(user_query)

    # print("📰 Live Web News (Playwright):")
    # for idx, item in enumerate(live_news, 1):
    #      print(f"{idx}. {item['title']} - {item['snippet']}\n")
    
    print("🧩 Expanded Sub-Queries (Phi-3-mini):")
    for idx, item in enumerate(sub_queries.get("sub_questions", []), 1):
        print(f"{idx}. {item}\n")
    return sub_queries