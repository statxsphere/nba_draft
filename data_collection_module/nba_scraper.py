from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

PAGE_SIZE_SELECT = r"/html/body/main/div/div/div[2]/div/div/nba-stat-table/div[1]/div/div/select"


class NbaScrape:
    def __init__(self):
        self.touches = pd.DataFrame()
        self.passes = pd.DataFrame()
        self.isolation = pd.DataFrame()
        self.transition = pd.DataFrame()
        self.drives = pd.DataFrame()
        self.pbh = pd.DataFrame()
        self.pbr = pd.DataFrame()
        self.cuts = pd.DataFrame()
        self.postup = pd.DataFrame()
        self.postdef = pd.DataFrame()
        self.peridef = pd.DataFrame()
        self.mergeddf = pd.DataFrame()
        self.years = list(range(2012, 2022))
        bas_url = f"https://www.nba.com/stats/players/traditional/?sort=PLAYER_NAME&dir=-1&Season={year - 1}-{str(year)[-2:]}&SeasonType=Regular%20Season"
        self.driver = webdriver.Firefox()
        self.driver.get(bas_url)
        element = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "I Accept")]')))
        element.click()

    def read_table(self):
        # show all rows, then parse the stats table
        select = Select(self.driver.find_element(By.XPATH, PAGE_SIZE_SELECT))
        select.select_by_index(0)
        src = self.driver.page_source
        soup = BeautifulSoup(src, "lxml")
        table = soup.find("div", attrs={"class": "nba-stat-table__overflow"})
        headers = table.find_all('th')
        headerlist = [h.text.strip() for h in headers]
        headerlist = [h.replace('\xa0', '') for h in headerlist]
        rows = table.find_all('tr')[1:]
        stats = [[td.get_text().strip() for td in row.find_all('td')] for row in rows]
        return pd.DataFrame(stats, columns=headerlist)

    def collect_touches(self, year):
        tch_url = f"https://www.nba.com/stats/players/touches/?Season={year-1}-{str(year)[-2:]}&SeasonType=Regular%20Season&sort=PLAYER_NAME&dir=-1"
        self.driver.get(tch_url)
        WebDriverWait(self.driver, 2).until(
            EC.element_to_be_clickable((By.XPATH, PAGE_SIZE_SELECT)))
        touches = self.read_table()
        self.touches = pd.concat([self.touches, touches])

    def collect_passes(self, year):
        pas_url = f"https://www.nba.com/stats/players/passing/?Season={year-1}-{str(year)[-2:]}&SeasonType=Regular%20Season&sort=PLAYER_NAME&dir=-1"
        self.driver.get(pas_url)
        WebDriverWait(self.driver, 2).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "I Accept")]')))
        passes = self.read_table()
        passes = passes.drop(columns=passes.columns[2:7])
        self.passes = pd.concat([self.passes, passes[['Player', 'Team', 'PassesMade', 'PassesReceived',
                                                      'AST', 'SecondaryAST', 'AST', 'ASTToPass%']]])

    def collect_isolation(self, year):
        iso_url = f"https://www.nba.com/stats/players/isolation/?sort=PLAYER_NAME&dir=-1&SeasonType=Regular Season&SeasonYear={year-1}-{str(year)[-2:]}"
        self.driver = webdriver.Firefox()
        self.driver.get(iso_url)
        element = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "I Accept")]')))
        element.click()
        passes = self.read_table()
        passes = passes.drop(columns=passes.columns[2:7])
        self.passes = pd.concat([self.passes, passes[['Player', 'Team', 'PassesMade', 'PassesReceived',
                                                      'AST', 'SecondaryAST', 'AST', 'ASTToPass%']]])

    def final_call(self):
        pass
