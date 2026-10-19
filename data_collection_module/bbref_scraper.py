from bs4 import BeautifulSoup
import requests
import pandas as pd


class BbScrape:
    def __init__(self):
        self.final_df = pd.DataFrame()
        self.years = list(range(2005, 2022))

    def collect_stats(self, year):
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
        headers = {'User-Agent': 'Mozilla/5.0'}
        source = requests.get(url, headers=headers)
        soup = BeautifulSoup(source.content, 'lxml')
        table = soup.find('table')
        head = table.find('thead').find_all('th')
        headerlist = [h.text.strip() for h in head]
        headerlist = [h.replace('\xa0', '') for h in headerlist][1:]
        rows = table.find_all('tr')[1:]
        stats = [[td.get_text().strip() for td in row.find_all('td')] for row in rows]
        df = pd.DataFrame(stats, columns=headerlist)
        self.final_df = pd.concat([self.final_df, df])

    def final_call(self):
        for i in self.years:
            print(f'Working on year: {i}..')
            self.collect_stats(i)
            print('Done!')
        return self.final_df


if __name__ == '__main__':
    scrape_obj = BbScrape()
    scrape_obj.final_call()
    scrape_obj.final_df.to_csv('data/bb_ref.csv', index=False)
