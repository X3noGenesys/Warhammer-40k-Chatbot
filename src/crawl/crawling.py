import os

from src.crawl.WebCrawlerAgent import WebCrawlerAgent
from src.crawl.WebCrawlerManager import WebCrawlerManager

if __name__ == "__main__":
    output_path = "data/crawled_data.jsonl"

    START_URLS = [
        "https://omnis-bibliotheca.com/index.php/L'Empereur_de_l'Humanité"
        ]

    BANNED_URLS = [
        "https://omnis-bibliotheca.com/index.php?title=",
        "https://omnis-bibliotheca.com/index.php/Accueil",
        "https://omnis-bibliotheca.com/index.php/Spécial",
        "https://omnis-bibliotheca.com/index.php/Fichier",
        "https://omnis-bibliotheca.com/index.php/Partenaires",
        "https://omnis-bibliotheca.com/index.php/Utilisateur",
        "https://omnis-bibliotheca.com/index.php/Liens",
        "https://omnis-bibliotheca.com/index.php/GW-Copyright",
        "https://omnis-bibliotheca.com/index.php/Omnis_Bibliotheca",
        "https://omnis-bibliotheca.com/index.php/Discussion",
        "https://omnis-bibliotheca.com/index.php/Catégorie",
        "https://omnis-bibliotheca.com/index.php/Aide",
        "https://omnis-bibliotheca.com/index.php/Modèle",
        "https://omnis-bibliotheca.com/cdn-cgi"
        ]
    
    USER_AGENT = "MyCrawler/1.0"
    TIMEOUT = 3600

    os.makedirs("data", exist_ok=True)
    os.remove(output_path) if os.path.exists(output_path) else None
    open(output_path, "w").close()


    manager = WebCrawlerManager(START_URLS, USER_AGENT, max_workers=8, timeout=TIMEOUT, banned_urls=BANNED_URLS)

    workers = [
        WebCrawlerAgent(manager, i)
        for i in range(manager.max_workers)
    ]

    for w in workers:
        w.start()

    for w in workers:
        w.join()

    print(f"\nPages crawled : {len(manager.data)}")