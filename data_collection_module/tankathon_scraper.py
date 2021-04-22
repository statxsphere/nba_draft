from bs4 import BeautifulSoup
import requests
import pandas as pd


class ScrapingModule:
    def __init__(self):
        self.years = list(range(2004, 2021))
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
        self.df = pd.DataFrame()

    def collect_singleyr_stats(self, year):
        file = requests.get(f"http://www.tankathon.com/past_drafts/{str(year)}").text
        soup = BeautifulSoup(file, 'lxml')
        i = 1
        # name
        for names in soup.find_all('div', class_='mock-row-name'):
            self.name.append(names.text)

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

    def set_years(self):
        input1 = int(input('Enter starting year: '))
        input2 = int(input('Enter ending year: '))
        self.years = list(range(input1, input2+1))

    def final_call(self):
        self.set_years()
        self.collect_all()
        statdict = {'name': self.name, 'position': self.position, 'pick': self.pick, 'height_cm': self.height,
                    'weight_lb': self.weight, 'c_year': self.c_year, 'age': self.age, 'points': self.points,
                    'reb': self.rebounds, 'ast': self.assists, 'TS': self.ts, 'usg': self.usg, 'o_bpm': self.obp,
                    'd_bpm': self.dbp}
        self.df = pd.DataFrame(statdict)
        return self.df
