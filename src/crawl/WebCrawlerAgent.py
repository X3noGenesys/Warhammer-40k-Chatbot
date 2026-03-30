import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import time
import threading
import trafilatura

class WebCrawlerAgent(threading.Thread):
    def __init__(self, manager, agent_id):
        super().__init__(daemon=True)
        self.manager = manager
        self.agent_id = agent_id
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": manager.user_agent
        })

    def run(self):
        while True:
            url = self.manager.get_next_url()
            if url is None:
                return

            if self.manager.already_visited(url):
                continue

            if self.manager.is_url_banned(url):
                print(f"🚫 [{self.agent_id}] Banned url {url}")
                continue

            if not self.manager.is_html_page(url):
                print(f"🚫 [{self.agent_id}] Non-HTML file {url}")
                continue

            if not self.manager.can_fetch(url):
                print(f"🚫 [{self.agent_id}] robots.txt blocks {url}")
                continue

            try:
                print(f"🔍 [{self.agent_id}] Crawl {url}")
                response = self.session.get(url, timeout=10)

                if response.status_code != 200:
                    continue

                self.manager.mark_visited(url)
                data, links = self.parse(url, response.text)
                self.manager.store_data(data)
                self.manager.add_urls(links)
    
                time.sleep(1)

            except Exception as e:
                print(f"⚠️ [{self.agent_id}] Erreur {url}: {e}")

    def parse(self, base_url, html):
        soup = BeautifulSoup(html, "html.parser")

        title = soup.title.string if soup.title and soup.title.string else ""

        text = trafilatura.extract(html) or ""

        links = set()

        for a in soup.find_all("a", href=True):
            raw_url = urljoin(base_url, a["href"])
            parsed = urlparse(raw_url)

            if parsed.scheme not in ("http", "https"):
                continue

            parsed = parsed._replace(fragment="")

            clean_url = urlunparse(parsed)

            links.add(clean_url)

        data = {
            "url": base_url,
            "title": title,
            "text": text
        }

        return data, list(links)