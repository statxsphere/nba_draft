from tankathon_scraper import ScrapingModule


if __name__ == "__main__":
    scrape_obj = ScrapingModule()
    finaldf = scrape_obj.final_call()
    print(finaldf)