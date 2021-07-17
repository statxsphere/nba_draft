from bs4 import BeautifulSoup
import requests
import pandas as pd


class BbScrape:
    def __init__(self):
        self.final_df = pd.DataFrame()
        self.years = [a for a in range(2012, 2022)]

    def collect_stats(self, year):
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
        headers = {'User-Agent': 'Mozilla/5.0'}
        source = requests.get(url, headers)
        soup = BeautifulSoup(source.content, 'lxml')
        table = soup.find('table')
        head = table.find('thead').findAll('th')
        headerlist = [h.text.strip() for h in head]
        headerlist = [h.replace('\xa0', '') for h in headerlist][1:]
        rows = table.findAll('tr')[1:]
        stats = [[td.getText().strip() for td in rows[i].findAll('td')] for i in range(len(rows))]
        self.final_df.append(pd.DataFrame(stats, columns=headerlist))

    def final_call(self):
        for i in self.years:
            self.collect_stats(i)


if __name__ == '__main__':
    scrape_obj = BbScrape()
    scrape_obj.final_call()
    scrape_obj.final_df.to_csv('data/bb_ref.csv', index=False)
