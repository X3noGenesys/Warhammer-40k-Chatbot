import threading
import queue
import urllib.robotparser as robotparser
from urllib.parse import urlparse, urljoin, quote, unquote
import time

class WebCrawlerManager:
    def __init__(self, start_urls, user_agent, max_workers=4, timeout=300, banned_urls=None):
        self.user_agent = user_agent
        self.frontier = queue.Queue()
        self.visited = set()
        self.visited_lock = threading.Lock()

        self.data = []
        self.data_lock = threading.Lock()

        self.robots = {}
        self.robots_lock = threading.Lock()

        self.allowed_domains = set(urlparse(url).netloc for url in start_urls)

        self.banned_urls = [self._normalize_url(url) for url in (banned_urls or [])]
        self.filename = "data/crawled_data.jsonl"
        self.start_time = time.time()
        self.timeout = timeout

        for url in start_urls:
            decoded_url = self._decode_url(url)
            self.frontier.put(decoded_url)

        self.max_workers = max_workers
        self.total_count = 0

    # --- Robots.txt ---
    def get_robot_parser(self, url):
        domain = urlparse(url).netloc

        with self.robots_lock:
            if domain in self.robots:
                return self.robots[domain]

            rp = robotparser.RobotFileParser()
            rp.set_url(f"https://{domain}/robots.txt")
            try:
                rp.read()
            except Exception:
                pass

            self.robots[domain] = rp
            return rp

    def can_fetch(self, url):
        rp = self.get_robot_parser(url)
        return rp.can_fetch(self.user_agent, url)

    def _normalize_url(self, url):
        decoded_url = unquote(url)
        parsed = urlparse(decoded_url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{quote(parsed.path, safe='/')}?{parsed.query}" if parsed.query else f"{parsed.scheme}://{parsed.netloc}{quote(parsed.path, safe='/')}"
        return normalized.rstrip('?')

    def _decode_url(self, url):
        return unquote(url)

    def is_url_banned(self, url):
        normalized_url = self._normalize_url(url)
        for banned_prefix in self.banned_urls:
            if normalized_url.startswith(banned_prefix):
                return True
        return False

    def is_html_page(self, url):
        ignored_extensions = {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt',
            '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2',
            '.exe', '.dll', '.sh', '.bat', '.com',
            '.mp4', '.avi', '.mov', '.mp3', '.wav', '.flac', '.mkv', '.m4v',
            '.woff', '.woff2', '.ttf', '.eot', '.otf',
            '.css', '.js', '.json', '.xml', '.rss'
        }
        
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        for ext in ignored_extensions:
            if path.endswith(ext):
                return False
        
        return True

    def get_next_url(self):
        if time.time() - self.start_time > self.timeout:
            return None
        try:
            return self.frontier.get(timeout=3)
        except queue.Empty:
            return None

    def add_urls(self, urls):
        for url in urls:
            url = self._decode_url(url)
            domain = urlparse(url).netloc
            if domain not in self.allowed_domains:
                continue
            if self.is_url_banned(url):
                continue
            if not self.is_html_page(url):
                continue
            with self.visited_lock:
                if url not in self.visited:
                    self.frontier.put(url)

    def mark_visited(self, url):
        with self.visited_lock:
            self.visited.add(url)

    def already_visited(self, url):
        with self.visited_lock:
            return url in self.visited

    def store_data(self, item):
        if "url" in item:
            item["url"] = self._decode_url(item["url"])
        with self.data_lock:
            self.data.append(item)
            self.total_count += 1

            import json
            with open(self.filename, "a", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

    def save_data_to_file(self, filename="crawled_data.json"):
        import json
        with self.data_lock:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)
        print(f"Données sauvegardées dans {filename}")