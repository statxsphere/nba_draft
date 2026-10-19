from bs4 import BeautifulSoup
import requests
import pandas as pd


def stat_value(stats, section, stat):
    try:
        return float(stats.find('div', class_=section).find('div', class_=stat).find('div').text)
    except Exception:
        return None


def to_cm(feet_inches):
    parts = feet_inches.replace('"', '').split("'")
    inches = int(parts[0]) * 12 + float(parts[1])
    return round(inches * 2.54, 1)


class TankathonScrape:
    def __init__(self):
        self.years = list(range(2011, 2019))
        self.name = []
        self.position = []
        self.pick = []
        self.height = []
        self.weight = []
        self.c_year = []
        self.age = []
        self.points = []
        self.rebounds = []
        self.assists = []
        self.steals = []
        self.blocks = []
        self.ts = []
        self.usg = []
        self.obp = []
        self.dbp = []
        self.links = []
        self.wing = []
        self.games = []
        self.mp = []
        self.ft = []
        self.tpp = []
        self.tpa = []
        self.fta = []
        self.df = pd.DataFrame()

    def parse_draft_page(self, soup):
        # name
        for names in soup.find_all('div', class_='mock-row-name'):
            self.name.append(names.text)

        # links
        for links in soup.find_all('a', class_='primary-hover'):
            if links.has_attr('href'):
                if 'player' in links['href']:
                    self.links.append(f"http://www.tankathon.com/{links['href']}")

        # position
        for positions in soup.find_all('div', class_='mock-row-school-position'):
            self.position.append(positions.text[0:2].rstrip())

        # measurements
        for measurements in soup.find_all('div', class_='section height-weight'):
            a = measurements.find_all('div')
            self.height.append(to_cm(a[0].text))
            self.weight.append(int(a[1].text[0:4]))

        # age
        for i in soup.find_all('div', class_='section year-age desktop'):
            x = i.find_all('div')
            self.c_year.append(x[0].text)
            self.age.append(float(x[1].text[0:4]))

        # game stats
        for stats in soup.find_all('div', class_='mock-row-stats'):
            self.points.append(stat_value(stats, 'stats-per-36', 'stat pts'))
            self.rebounds.append(stat_value(stats, 'stats-per-36', 'stat reb'))
            self.assists.append(stat_value(stats, 'stats-per-36', 'stat ast'))
            self.steals.append(stat_value(stats, 'stats-per-36', 'stat stl desktop'))
            self.blocks.append(stat_value(stats, 'stats-per-36', 'stat blk desktop'))
            self.ts.append(stat_value(stats, 'stats-advanced', 'stat pts'))
            self.usg.append(stat_value(stats, 'stats-advanced', 'stat reb'))
            self.obp.append(stat_value(stats, 'stats-advanced', 'stat ast'))
            self.dbp.append(stat_value(stats, 'stats-advanced', 'stat blk desktop'))

    def collect_singleyr_stats(self, year):
        file = requests.get(f"http://www.tankathon.com/past_drafts/{str(year)}").text
        soup = BeautifulSoup(file, 'lxml')
        self.parse_draft_page(soup)

        # pick
        for i in range(1, 61):
            self.pick.append(i)

    def collect_all(self):
        for year in self.years:
            print(f'collecting stats for year {year}..')
            self.collect_singleyr_stats(year)
            print('Done! Moving On.')

    def collect_from_links(self):
        i = 1
        for link in self.links:
            print(f'starting player{i}..')
            f = requests.get(link).text
            s = BeautifulSoup(f, 'lxml')
            ft = None
            t3p = None
            p3a = None
            fta = None
            wing = None
            g = None
            mp = None
            mp1 = []

            for stat in s.find_all('div', class_='stat-container'):
                if stat.find('div', class_='stat-label').text == "FT%":
                    ft = stat.find('div', class_='stat-data').text
                if stat.find('div', class_='stat-label').text == "3P%":
                    t3p = stat.find('div', class_='stat-data').text
                if "3PA" in stat.find('div', class_='stat-label').text:
                    p3a = stat.find('div', class_='stat-data').text
                if "FTA" in stat.find('div', class_='stat-label').text:
                    fta = stat.find('div', class_='stat-data').text
                if stat.find('div', class_='stat-label').text == "G":
                    g = stat.find('div', class_='stat-data').text
                if stat.find('div', class_='stat-label').text == "MP":
                    mp1.append(stat.find('div', class_='stat-data').text)

            self.ft.append(ft)
            self.tpp.append(t3p)
            self.tpa.append(p3a)
            self.fta.append(fta)
            self.games.append(g)
            if mp1:
                self.mp.append(mp1[0])
            else:
                self.mp.append(mp)

            for stat in s.find_all('span'):
                if 'wingspan' in stat.text:
                    wing = stat.text[1:-9]
            if wing:
                self.wing.append(to_cm(wing))
            else:
                self.wing.append(wing)
            print('done')
            i += 1

    def make_df(self):
        statdict = {'name': self.name, 'position': self.position, 'pick': self.pick, 'height_cm': self.height,
                    'wingspan_cm': self.wing, 'weight_lb': self.weight, 'c_year': self.c_year, 'age': self.age,
                    'games': self.games, 'mp': self.mp, 'points': self.points, 'reb': self.rebounds,
                    'ast': self.assists, 'steals': self.steals, 'blocks': self.blocks, 'TS': self.ts,
                    'usg': self.usg, 'o_bpm': self.obp, 'd_bpm': self.dbp, '3p%': self.tpp, 'ft%': self.ft,
                    '3pa': self.tpa, 'fta': self.fta}
        self.df = pd.DataFrame(statdict)
        return self.df

    def scrape_link(self, x):
        file = requests.get(x).text
        soup = BeautifulSoup(file, 'lxml')
        self.parse_draft_page(soup)

        # pick
        for i in range(1, len(self.name) + 1):
            self.pick.append(i)

        self.collect_from_links()
        self.make_df()

    def set_years(self):
        input1 = int(input('Enter starting year: '))
        input2 = int(input('Enter ending year: '))
        if input2 != -1:
            self.years = list(range(input1, input2 + 1))
        else:
            self.years = [input1]

    def final_call(self):
        self.set_years()
        self.collect_all()
        print(f'no of. players is {len(self.name)}')
        print('starting link collection')
        self.collect_from_links()
        return self.make_df()


if __name__ == "__main__":
    scrape_obj = TankathonScrape()
    finaldf = scrape_obj.final_call()
    finaldf.to_csv('data/tankathon.csv', index=False)
