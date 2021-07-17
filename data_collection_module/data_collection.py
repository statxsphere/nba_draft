from tankathon_scraper import TankathonScrape
from bbref_scraper import BbScrape
from nba_scraper import NbaScrape


if __name__ == "__main__":
    # tankathon
    scrape_obj = TankathonScrape()
    df = scrape_obj.final_call()
    df.to_csv('data/tankathon.csv', index=False)

    # basketball ref
    scrape_obj = BbScrape()
    df = scrape_obj.final_call()
    df.to_csv('data/bbref.csv', index=False)

    # nba.com
    scrape_obj = NbaScrape()
    df = scrape_obj.final_call()
    df.to_csv('data/nbacom.csv', index=False)
