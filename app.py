from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, request, jsonify, send_file
import feedparser
import requests
import os
import io
import re
import json
import google.generativeai as genai
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
 
app = Flask(__name__)
 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
 
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment. "
          "AI Summary and Sentiment features will not work until it is set.")
    gemini_model = None
else:
    genai.configure(api_key=GEMINI_API_KEY)
    # Current GA (generally available) model as of July 2026.
    # gemini-2.5-flash was retired for new users; gemini-3.5-flash-lite
    # is the fastest, lowest-cost option in the current lineup.
    gemini_model = genai.GenerativeModel("gemini-3.5-flash-lite")
 
# NOTE: the legacy `google-generativeai` SDK's GenerationConfig does NOT
# support "thinking_level" / "thinking_config" at all (that's only in the
# newer `google-genai` SDK). Passing it -- nested or not -- raises:
#   "Unknown field for GenerationConfig: thinking_config"
# gemini-3.5-flash-lite is already fast by default, so we just omit it.
FAST_GENERATION_CONFIG = None
 
RSS_FEEDS = {
    "india": "https://www.news18.com/rss/india.xml",
    "cricket": "https://www.news18.com/rss/cricketnext.xml",
    "world": "https://www.news18.com/rss/world.xml"
}
 
DEFAULT_CATEGORY = "india"
 
WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
 
HEADERS = {
    "User-Agent": "NewsChatbot/1.0 (contact: your_email@example.com)"
}
 
 
def get_category_articles(category, limit=10):
    """Fetch top articles for a given RSS category."""
    feed_url = RSS_FEEDS.get(category, RSS_FEEDS["india"])
    feed = feedparser.parse(feed_url)
    articles = []
    for entry in feed.entries[:limit]:
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "summary": entry.summary
        })
    return articles
 
 
def get_person_summary(name):
    """Fetch a precise one-paragraph summary about a person from Wikipedia."""
    try:
        resp = requests.get(
            WIKI_SUMMARY_URL.format(name.replace(" ", "_")),
            headers=HEADERS,
            timeout=5
        )
 
        if resp.status_code != 200:
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": name,
                "format": "json",
                "srlimit": 1
            }
            search_resp = requests.get(
                WIKI_SEARCH_URL,
                params=search_params,
                headers=HEADERS,
                timeout=5
            ).json()
            results = search_resp.get("query", {}).get("search", [])
            if not results:
                return None
            exact_title = results[0]["title"]
            resp = requests.get(
                WIKI_SUMMARY_URL.format(exact_title.replace(" ", "_")),
                headers=HEADERS,
                timeout=5
            )
            if resp.status_code != 200:
                return None
 
        data = resp.json()
 
        if data.get("type") == "disambiguation":
            return None
 
        return {
            "title": data.get("title"),
            "summary": data.get("extract"),
            "thumbnail": data.get("thumbnail", {}).get("source") if data.get("thumbnail") else None,
            "link": data.get("content_urls", {}).get("desktop", {}).get("page")
        }
    except (requests.RequestException, ValueError):
        return None
 
 
def get_person_news(name, limit=5):
    """Fetch latest news articles related to the person using Google News RSS search."""
    try:
        query = requests.utils.quote(name)
        feed_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(feed_url)
 
        news_list = []
        for entry in feed.entries[:limit]:
            news_list.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.get("published", "")
            })
        return news_list
    except Exception:
        return []
 
 
def summarize_text(text, title=None):
    """Summarize a single news article/snippet using the Gemini API."""
    if not gemini_model:
        print("SUMMARIZE ERROR: GEMINI_API_KEY is not set.")
        return None
 
    if not text or not text.strip():
        print("SUMMARIZE ERROR: empty text was passed in.")
        return None
 
    prompt = f"""Summarize the following news content in 2-3 concise sentences.
Focus on key facts only. Do not add opinions or information not present in the text.
 
{f"Title: {title}" if title else ""}
 
Content:
{text}
"""
    try:
        if FAST_GENERATION_CONFIG:
            response = gemini_model.generate_content(prompt, generation_config=FAST_GENERATION_CONFIG)
        else:
            response = gemini_model.generate_content(prompt)
 
        # If the model returns no candidates (e.g. blocked by safety filters),
        # response.text raises instead of just being empty -- guard for that.
        if not response.candidates:
            print("SUMMARIZE ERROR: no candidates returned. "
                  "prompt_feedback:", getattr(response, "prompt_feedback", None))
            return None
 
        summary = response.text.strip()
        if not summary:
            print("SUMMARIZE ERROR: model returned an empty string.")
            return None
 
        return summary
    except Exception as e:
        print("SUMMARIZE ERROR:", repr(e))
        return None
 
 
def analyze_sentiment(text, title=None):
    """Classify the sentiment of a news article/snippet using the Gemini API.
 
    Returns a dict like {"sentiment": "Positive", "reason": "..."} or None on failure.
    """
    if not gemini_model:
        print("SENTIMENT ERROR: GEMINI_API_KEY is not set.")
        return None
 
    if not text or not text.strip():
        print("SENTIMENT ERROR: empty text was passed in.")
        return None
 
    prompt = f"""Analyze the overall sentiment/tone of the following news content.
Respond with ONLY a valid JSON object and nothing else (no markdown, no code fences, no extra text).
Use exactly this format:
{{"sentiment": "Positive", "reason": "one short sentence explaining why"}}
 
The "sentiment" value must be exactly one of: "Positive", "Negative", "Neutral".
 
{f"Title: {title}" if title else ""}
 
Content:
{text}
"""
    try:
        if FAST_GENERATION_CONFIG:
            response = gemini_model.generate_content(prompt, generation_config=FAST_GENERATION_CONFIG)
        else:
            response = gemini_model.generate_content(prompt)
 
        if not response.candidates:
            print("SENTIMENT ERROR: no candidates returned. "
                  "prompt_feedback:", getattr(response, "prompt_feedback", None))
            return None
 
        raw = response.text.strip()
 
        # Strip accidental markdown code fences if the model adds them anyway
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
 
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Model didn't return clean JSON -- try to salvage just the
            # {...} portion in case it added stray text around it.
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                print("SENTIMENT ERROR: could not parse JSON from ->", repr(raw))
                return None
            parsed = json.loads(match.group(0))
 
        raw_sentiment = str(parsed.get("sentiment", "")).strip()
        reason = str(parsed.get("reason", "")).strip()
 
        # Normalize: lowercase + strip punctuation before matching, so things
        # like "positive.", "POSITIVE", " Positive " all still match.
        normalized = re.sub(r"[^a-z]", "", raw_sentiment.lower())
 
        sentiment_map = {
            "positive": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
        }
        sentiment = sentiment_map.get(normalized)
 
        if not sentiment:
            print("SENTIMENT ERROR: unexpected sentiment value ->", repr(raw))
            return None
 
        return {"sentiment": sentiment, "reason": reason}
    except Exception as e:
        print("SENTIMENT ERROR:", repr(e))
        return None
 
 
def generate_pdf(title, content, link=None):
    """Generate a simple PDF with a title, body content, and an optional link."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    story = []
 
    safe_title = escape(title or "Summary")
    story.append(Paragraph(safe_title, styles["Title"]))
    story.append(Spacer(1, 14))
 
    safe_content = escape(content or "").replace("\n", "<br/>")
    story.append(Paragraph(safe_content, styles["Normal"]))
 
    if link:
        safe_link = escape(link)
        story.append(Spacer(1, 14))
        story.append(Paragraph(f'<link href="{safe_link}">{safe_link}</link>', styles["Normal"]))
 
    doc.build(story)
    buffer.seek(0)
    return buffer
 
 
@app.route("/", methods=["GET", "POST"])
def index():
    articles = []
    person_result = None
    person_news = []
    error = None
    search_query = ""
    selected_category = DEFAULT_CATEGORY
 
    if request.method == "POST":
        if "person" in request.form:
            search_query = request.form.get("person", "").strip()
            if search_query:
                person_result = get_person_summary(search_query)
                person_news = get_person_news(search_query)
                if not person_result:
                    error = f"'{search_query}' ke baare mein koi precise jaankari nahi mili."
        elif "category" in request.form:
            selected_category = request.form.get("category", DEFAULT_CATEGORY)
            articles = get_category_articles(selected_category)
    else:
        # GET request (page load / refresh): auto-load today's top news
        # for the default category, without the user clicking anything.
        articles = get_category_articles(DEFAULT_CATEGORY)
 
    return render_template(
        "index.html",
        articles=articles,
        person_result=person_result,
        person_news=person_news,
        error=error,
        search_query=search_query,
        selected_category=selected_category
    )
 
 
@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True) or {}
    title = data.get("title", "")
    text = data.get("summary", "")
 
    print(f"SUMMARIZE REQUEST: title={title!r} text_len={len(text or '')}")
 
    ai_summary = summarize_text(text, title)
 
    if not ai_summary:
        return jsonify({"success": False, "error": "Summary generate nahi ho paya."}), 500
 
    return jsonify({"success": True, "summary": ai_summary})
 
 
@app.route("/sentiment", methods=["POST"])
def sentiment():
    data = request.get_json(silent=True) or {}
    title = data.get("title", "")
    text = data.get("summary", "")
 
    print(f"SENTIMENT REQUEST: title={title!r} text_len={len(text or '')}")
 
    result = analyze_sentiment(text, title)
 
    if not result:
        return jsonify({"success": False, "error": "Sentiment analyze nahi ho paya."}), 500
 
    return jsonify({"success": True, "sentiment": result["sentiment"], "reason": result["reason"]})
 
 
@app.route("/download-pdf", methods=["POST"])
def download_pdf():
    data = request.get_json()
    title = (data.get("title") or "Summary").strip()
    content = (data.get("content") or "").strip()
    link = (data.get("link") or "").strip()
 
    if not content:
        return jsonify({"success": False, "error": "Content khaali hai, PDF nahi banaya ja sakta."}), 400
 
    pdf_buffer = generate_pdf(title, content, link)
 
    safe_name = re.sub(r'[^A-Za-z0-9]+', '_', title)[:50].strip('_') or "summary"
    filename = f"{safe_name}.pdf"
 
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename
    )
 
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)