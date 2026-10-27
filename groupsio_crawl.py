"""groups.io topic-list crawler for the Qdrant incremental update.

Drop-in replacement for crawl_new_topics / extract_topic_info / extract_next_page_info in
incremental_update_qdrant.py.  Differences from the old versions:
  * waits for topic links instead of the `table.table-condensed` class (no longer present)
  * falls back to a markup-independent extraction if the old row structure has changed
  * raises CrawlError (instead of printing and returning nothing) when a page does not load,
    parses to zero topics, or has no dates.  The caller must NOT save state in that case.
  * on failure prints the page url/title/text and saves groupsio_failure.png / .html
  * with no cutoff date every topic counts as new (the old code silently collected nothing)
"""
import asyncio
import os
import re
from datetime import datetime, timedelta
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from dateutil import parser
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

BASE_GROUP_URL = "https://groups.io/g/Hipresurfacingsite"
BASE_URL = f"{BASE_GROUP_URL}/topics"
TOPIC_PATH = urlparse(BASE_GROUP_URL).path.rstrip("/") + "/topic/"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

_MONTH = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


class CrawlError(Exception):
    """The topic list could not be fetched or understood. Never treat this as 'no new topics'."""


# ----------------------------------------------------------------------------- helpers
def _date_str(text):
    """Return YYYY-MM-DD for a tooltip / datetime attribute, or None if it is not a date."""
    if not text:
        return None
    text = text.strip()
    for fmt in ("%b %d, %Y %I:%M%p", "%b %d, %Y %I:%M %p", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    looks_like_date = (_YEAR.search(text) and _MONTH.search(text)) or re.match(r"^\d{4}-\d{2}-\d{2}", text)
    if looks_like_date:
        try:
            return parser.parse(text).strftime("%Y-%m-%d")
        except Exception:
            return None
    return None


def _abs_url(href, base_group_url=BASE_GROUP_URL):
    if href.startswith("http"):
        return href
    p = urlparse(base_group_url)
    if not href.startswith("/"):
        href = "/" + href
    if href.startswith(p.path.rstrip("/") + "/"):          # '/g/Group/topic/...'
        return f"{p.scheme}://{p.netloc}{href}"
    return base_group_url.rstrip("/") + href               # old style: '/topic/...'


def _dates_in(container):
    """All parseable dates in title= and datetime= attributes inside a container."""
    found = []
    for el in container.find_all(True):
        for attr in ("title", "datetime"):
            d = _date_str(el.get(attr, ""))
            if d:
                found.append(d)
    return sorted(found)


# ----------------------------------------------------------------------------- extraction
def _primary_topics(soup, base_group_url):
    """The original row structure: <tr><td> a.subject + span.thread-attribution."""
    topics = []
    for row in soup.find_all("tr"):
        topic_cell = row.find("td")
        if not topic_cell:
            continue
        subject_link = topic_cell.find("a", class_="subject")
        if not subject_link or not subject_link.get("href"):
            continue
        started = recent = None
        thread_attr = topic_cell.find("span", class_="thread-attribution")
        if thread_attr:
            spans = thread_attr.find_all("span", title=True)
            if spans:
                started = _date_str(spans[0].get("title", ""))
                if len(spans) > 1:
                    recent = _date_str(spans[-1].get("title", ""))
        topics.append({
            "url": _abs_url(subject_link["href"], base_group_url),
            "subject": subject_link.get_text(strip=True),
            "started_date": started,
            "most_recent_date": recent,
        })
    return topics


def _fallback_topics(soup, base_group_url):
    """Markup-independent: any link into <group>/topic/..., dates from title=/datetime= attributes."""
    best = {}                                    # url -> (text length, <a>)
    for a in soup.find_all("a", href=re.compile(re.escape(TOPIC_PATH))):
        url = _abs_url(a["href"], base_group_url).split("#")[0].split("?")[0]
        text = a.get_text(strip=True)
        if not text or re.fullmatch(r"[\d,\s]+", text):
            continue                             # e.g. a "12" message-count link
        if url not in best or len(text) > best[url][0]:
            best[url] = (len(text), a)

    topics = []
    for url, (_, a) in best.items():
        container = a.find_parent("tr") or a.find_parent("li") or a.parent.parent or a.parent
        dates = _dates_in(container)
        topics.append({
            "url": url,
            "subject": a.get_text(strip=True),
            "started_date": dates[0] if dates else None,
            "most_recent_date": dates[-1] if dates else None,
        })
    return topics


async def extract_topic_info(html: str, base_group_url: str = BASE_GROUP_URL):
    soup = BeautifulSoup(html, "html.parser")
    topics = _primary_topics(soup, base_group_url)
    if topics:
        return topics
    topics = _fallback_topics(soup, base_group_url)
    if topics:
        print(f"  [INFO] row structure not found; used link-based fallback ({len(topics)} topics)")
    return topics


async def extract_next_page_info(html: str):
    """Return the 'after' token for the next page, or ''."""
    soup = BeautifulSoup(html, "html.parser")
    pagination = soup.find("ul", class_="pagination")
    if pagination:
        chevron = pagination.find("i", class_="fa-chevron-right")
        link = chevron.find_parent("a") if chevron else None
        if link and link.get("href"):
            m = re.search(r"after=([^&]+)", link["href"])
            if m:
                return m.group(1)
    for a in soup.find_all("a", href=re.compile(r"[?&]after=")):      # fallback
        label = " ".join(a.get_text().split()).lower()
        is_next = ("next" in label or a.get("rel") == ["next"]
                   or a.find("i", class_=re.compile(r"(chevron|angle)-right")))
        if is_next:
            m = re.search(r"after=([^&]+)", a["href"])
            if m:
                return m.group(1)
    return ""


# ----------------------------------------------------------------------------- crawling
async def _dump(page, stem="groupsio_failure"):
    """Print what the page really was and save a screenshot + HTML for the workflow artifact."""
    try:
        title = await page.title()
        raw = await page.evaluate("document.body ? document.body.innerText : ''")
        print(f"[DIAG] url={page.url} title={title!r}")
        print(f"[DIAG] text: {' '.join(raw.split())[:300]}")
        await page.screenshot(path=f"{stem}.png", full_page=True)
        with open(f"{stem}.html", "w", encoding="utf-8") as f:
            f.write(await page.content())
    except Exception as e:
        print(f"[DIAG] could not capture diagnostics: {e}")


async def crawl_new_topics(latest_date_in_qdrant, debug=False):
    """Crawl topics newer than the cutoff. Raises CrawlError if the crawl cannot be trusted.

    CUTOFF_MARGIN_DAYS (env, default 0) widens the cutoff to catch same-day posts. Only set it
    above 0 once you know re-ingesting a topic overwrites its old points instead of duplicating.
    """
    margin = int(os.environ.get("CUTOFF_MARGIN_DAYS", "0") or 0)
    cutoff = None
    if latest_date_in_qdrant:
        cutoff = parser.parse(latest_date_in_qdrant) - timedelta(days=margin)
    new_topics = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            context = await browser.new_context(user_agent=USER_AGENT)
            page = await context.new_page()

            current_url, page_num, seen_tokens = BASE_URL, 1, set()
            while True:
                print(f"\nFetching page {page_num}: {current_url}")
                await page.goto(current_url, wait_until="domcontentloaded", timeout=60000)
                try:
                    await page.wait_for_selector(f'a[href*="{TOPIC_PATH}"]',
                                                 state="attached", timeout=30000)
                except PlaywrightTimeoutError as e:
                    await _dump(page)
                    raise CrawlError(f"page {page_num}: no topic links appeared") from e

                html = await page.content()
                page_topics = await extract_topic_info(html, BASE_GROUP_URL)
                if not page_topics:
                    await _dump(page)
                    raise CrawlError(f"page {page_num} loaded but 0 topics could be parsed")

                undated = [t for t in page_topics
                           if not (t["most_recent_date"] or t["started_date"])]
                if len(undated) > 0.2 * len(page_topics):
                    await _dump(page)
                    raise CrawlError(f"page {page_num}: {len(undated)}/{len(page_topics)} "
                                     "topics have no date (date markup changed?)")

                page_has_new = False
                for t in page_topics:
                    if cutoff is None:
                        is_new = True
                    else:
                        recent = parser.parse(t["most_recent_date"]) if t["most_recent_date"] else None
                        started = parser.parse(t["started_date"]) if t["started_date"] else None
                        is_new = bool((recent and recent > cutoff) or (started and started > cutoff))
                    if is_new:
                        new_topics.append(t)
                        page_has_new = True

                print(f"Found {len(page_topics)} topics on page {page_num}, "
                      f"{len(new_topics)} new so far.")
                if cutoff is not None and not page_has_new:
                    print(f"No new topics on page {page_num}. Stopping crawl.")
                    break

                token = await extract_next_page_info(html)
                if not token or token in seen_tokens:
                    print("No more pages to fetch")
                    break
                seen_tokens.add(token)
                page_num += 1
                current_url = f"{BASE_URL}?page={page_num}&after={token}"
                await asyncio.sleep(1)
        finally:
            await browser.close()

    return new_topics
