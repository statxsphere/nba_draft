from tankathon_scraper import TScrapingModule


if __name__ == "__main__":
    scrape_obj = TScrapingModule()
    finaldf = scrape_obj.final_call()
    finaldf.to_csv(f'data/{scrape_obj.years[0]}-{scrape_obj.years[-1]}.csv', index=False)