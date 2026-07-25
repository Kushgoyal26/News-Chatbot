from flask import Flask, render_template, request
import feedparser
import requests

app = Flask(__name__)

RSS_FEEDS = {
    "india": "https://www.news18.com/rss/india.xml",
    "cricket": "https://www.news18.com/rss/cricketnext.xml",
    "world": "https://www.news18.com/rss/world.xml"
}

WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "NewsChatbot/1.0 (contact: your_email@example.com)"
}


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


@app.route("/", methods=["GET", "POST"])
def index():
    articles = []
    person_result = None
    person_news = []
    error = None
    search_query = ""

    if request.method == "POST":
        if "person" in request.form:
            search_query = request.form.get("person", "").strip()
            if search_query:
                person_result = get_person_summary(search_query)
                person_news = get_person_news(search_query)
                if not person_result:
                    error = f"'{search_query}' ke baare mein koi precise jaankari nahi mili."
        elif "category" in request.form:
            category = request.form.get("category")
            feed_url = RSS_FEEDS.get(category, RSS_FEEDS["india"])
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                articles.append({
                    "title": entry.title,
                    "link": entry.link,
                    "summary": entry.summary
                })

    return render_template(
        "index.html",
        articles=articles,
        person_result=person_result,
        person_news=person_news,
        error=error,
        search_query=search_query
    )


if __name__ == "__main__":
    app.run(debug=True)