from bs4 import BeautifulSoup
import requests
import pandas as pd


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

    def collect_singleyr_stats(self, year):
        file = requests.get(f"http://www.tankathon.com/past_drafts/{str(year)}").text
        soup = BeautifulSoup(file, 'lxml')
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

        # pick
        for i in range(1, 61):
            self.pick.append(i)

        # measurements
        for measurements in soup.find_all('div', class_='section height-weight'):
            a = measurements.find_all('div')
            heights = a[0].text.replace('"', '').split("'")
            heights1 = int(heights[0]) * 12 + float(heights[1])
            self.height.append(round(heights1 * 2.54, 1))
            self.weight.append(int(a[1].text[0:4]))

        # age
        for i in soup.find_all('div', class_='section year-age desktop'):
            x = i.find_all('div')
            self.c_year.append(x[0].text)
            self.age.append(float(x[1].text[0:4]))

        # game stats
        for stats in soup.find_all('div', class_='mock-row-stats'):
            try:
                match_36 = stats.find('div', class_='stats-per-36')
                point = match_36.find('div', class_='stat pts')
                match1 = point.find('div')
                self.points.append(float(match1.text))
            except:
                self.points.append(None)

            try:
                match_36 = stats.find('div', class_='stats-per-36')
                reb = match_36.find('div', class_='stat reb')
                match2 = reb.find('div')
                self.rebounds.append(float(match2.text))
            except:
                self.rebounds.append(None)

            try:
                match_36 = stats.find('div', class_='stats-per-36')
                ast = match_36.find('div', class_='stat ast')
                match3 = ast.find('div')
                self.assists.append(float(match3.text))
            except:
                self.assists.append(None)

            try:
                match_36 = stats.find('div', class_='stats-per-36')
                stl = match_36.find('div', class_='stat stl desktop')
                match4 = stl.find('div')
                self.steals.append(float(match4.text))
            except:
                self.steals.append(None)

            try:
                match_36 = stats.find('div', class_='stats-per-36')
                blk = match_36.find('div', class_='stat blk desktop')
                match5 = blk.find('div')
                self.blocks.append(float(match5.text))
            except:
                self.blocks.append(None)

            try:
                match_adv = stats.find('div', class_='stats-advanced')
                TSp = match_adv.find('div', class_='stat pts')
                match6 = TSp.find('div')
                self.ts.append(float(match6.text))
            except:
                self.ts.append(None)

            try:
                match_adv = stats.find('div', class_='stats-advanced')
                usage = match_adv.find('div', class_='stat reb')
                match7 = usage.find('div')
                self.usg.append(float(match7.text))
            except:
                self.usg.append(None)

            try:
                match_adv = stats.find('div', class_='stats-advanced')
                bpo = match_adv.find('div', class_='stat ast')
                match8 = bpo.find('div')
                self.obp.append(float(match8.text))
            except:
                self.obp.append(None)

            try:
                match_adv = stats.find('div', class_='stats-advanced')
                bpd = match_adv.find('div', class_='stat blk desktop')
                match9 = bpd.find('div')
                self.dbp.append(float(match9.text))
            except:
                self.dbp.append(None)

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
                    mp = stat.find('div', class_='stat-data').text

            self.ft.append(ft)
            self.tpp.append(t3p)
            self.tpa.append(p3a)
            self.fta.append(fta)
            self.games.append(g)
            self.mp.append(mp)

            for stat in s.find_all('span'):
                if 'wingspan' in stat.text:
                    wing = stat.text[1:-9]
            if wing:
                wing1 = wing.replace('"', '').split("'")
                wing2 = int(wing1[0]) * 12 + float(wing1[1])
                self.wing.append(round(wing2 * 2.54, 1))
            else:
                self.wing.append(wing)
            print('done')
            i += 1

    def scrape_link(self, x):
        file = requests.get(x).text
        soup = BeautifulSoup(file, 'lxml')
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

        # pick
        for i in range(1, len(self.name) + 1):
            self.pick.append(i)

        # measurements
        for measurements in soup.find_all('div', class_='section height-weight'):
            a = measurements.find_all('div')
            heights = a[0].text.replace('"', '').split("'")
            heights1 = int(heights[0]) * 12 + float(heights[1])
            self.height.append(round(heights1 * 2.54, 1))
            self.weight.append(int(a[1].text[0:4]))

        # age
        for i in soup.find_all('div', class_='section year-age desktop'):
            x = i.find_all('div')
            self.c_year.append(x[0].text)
            self.age.append(float(x[1].text[0:4]))

        # game stats
        for stats in soup.find_all('div', class_='mock-row-stats'):
            try:
                match_36 = stats.find('div', class_='stats-per-36')
                point = match_36.find('div', class_='stat pts')
                match1 = point.find('div')
                self.points.append(float(match1.text))
            except:
                self.points.append(None)

            try:
                match_36 = stats.find('div', class_='stats-per-36')
                reb = match_36.find('div', class_='stat reb')
                match2 = reb.find('div')
                self.rebounds.append(float(match2.text))
            except:
                self.rebounds.append(None)

            try:
                match_36 = stats.find('div', class_='stats-per-36')
                ast = match_36.find('div', class_='stat ast')
                match3 = ast.find('div')
                self.assists.append(float(match3.text))
            except:
                self.assists.append(None)

            try:
                match_36 = stats.find('div', class_='stats-per-36')
                stl = match_36.find('div', class_='stat stl desktop')
                match4 = stl.find('div')
                self.steals.append(float(match4.text))
            except:
                self.steals.append(None)

            try:
                match_36 = stats.find('div', class_='stats-per-36')
                blk = match_36.find('div', class_='stat blk desktop')
                match5 = blk.find('div')
                self.blocks.append(float(match5.text))
            except:
                self.blocks.append(None)

            try:
                match_adv = stats.find('div', class_='stats-advanced')
                TSp = match_adv.find('div', class_='stat pts')
                match6 = TSp.find('div')
                self.ts.append(float(match6.text))
            except:
                self.ts.append(None)

            try:
                match_adv = stats.find('div', class_='stats-advanced')
                usage = match_adv.find('div', class_='stat reb')
                match7 = usage.find('div')
                self.usg.append(float(match7.text))
            except:
                self.usg.append(None)

            try:
                match_adv = stats.find('div', class_='stats-advanced')
                bpo = match_adv.find('div', class_='stat ast')
                match8 = bpo.find('div')
                self.obp.append(float(match8.text))
            except:
                self.obp.append(None)

            try:
                match_adv = stats.find('div', class_='stats-advanced')
                bpd = match_adv.find('div', class_='stat blk desktop')
                match9 = bpd.find('div')
                self.dbp.append(float(match9.text))
            except:
                self.dbp.append(None)

        self.collect_from_links()
        statdict = {'name': self.name, 'position': self.position, 'pick': self.pick, 'height_cm': self.height,
                    'wingspan_cm': self.wing, 'weight_lb': self.weight, 'c_year': self.c_year, 'age': self.age,
                    'games': self.games, 'mp': self.mp, 'points': self.points, 'reb': self.rebounds,
                    'ast': self.assists,
                    'steals': self.steals, 'blocks': self.blocks, 'TS': self.ts, 'usg': self.usg, 'o_bpm': self.obp,
                    'd_bpm': self.dbp, '3p%': self.tpp, 'ft%': self.ft, '3pa': self.tpa, 'fta': self.fta}

        self.df = pd.DataFrame(statdict)

    def set_years(self):
        input1 = int(input('Enter starting year: '))
        input2 = int(input('Enter ending year: '))
        if not input2 == -1:
            self.years = list(range(input1, input2+1))
        else:
            self.years = [input1]

    def final_call(self):
        self.set_years()
        self.collect_all()
        print(f'no of. players is {len(self.name)}')
        print('starting link collection')
        self.collect_from_links()
        statdict = {'name': self.name, 'position': self.position, 'pick': self.pick, 'height_cm': self.height,
                    'wingspan_cm': self.wing, 'weight_lb': self.weight, 'c_year': self.c_year, 'age': self.age,
                    'games': self.games, 'mp':self.mp, 'points': self.points, 'reb': self.rebounds, 'ast': self.assists,
                    'steals': self.steals, 'blocks':self.blocks, 'TS': self.ts, 'usg': self.usg, 'o_bpm': self.obp,
                    'd_bpm': self.dbp, '3p%': self.tpp, 'ft%': self.ft, '3pa': self.tpa, 'fta': self.fta}

        self.df = pd.DataFrame(statdict)
        return self.df


if __name__ == "__main__":
    scrape_obj = TankathonScrape()
    finaldf = scrape_obj.final_call()
    finaldf.to_csv(f'data/tankathon.csv', index=False)