from bs4 import BeautifulSoup
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


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
        self.years = [a for a in range(2012, 2022)]

    def collect_touches(self, year):
        tch_url = f"https://www.nba.com/stats/players/touches/?Season={year-1}-{str(year)[-2:]}&SeasonType=Regular%20Season&sort=PLAYER_NAME&dir=-1"
        driver = webdriver.Firefox()
        driver.get(tch_url)
        element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "I Accept")]')))
        element.click()
        select = Select(driver.find_element_by_xpath(
            r"/html/body/main/div/div/div[2]/div/div/nba-stat-table/div[1]/div/div/select"))
        select.select_by_index(0)
        src = driver.page_source
        soup = BeautifulSoup(src, "lxml")
        table = soup.find("div", attrs={"class": "nba-stat-table__overflow"})
        headers = table.findAll('th')
        headerlist = [h.text.strip() for h in headers]
        headerlist = [h.replace('\xa0', '') for h in headerlist]
        rows = table.findAll('tr')[1:]
        stats = [[td.getText().strip() for td in rows[i].findAll('td')] for i in range(len(rows))]
        touches = pd.DataFrame(stats, columns=headerlist)
        touches.drop(columns=touches.columns[2:7], inplace=True)
        self.touches = self.touches.append(touches)
        del tch_url, touches

    def collect_passes(self, year):
        pas_url = f"https://www.nba.com/stats/players/passing/?Season={year-1}-{str(year)[-2:]}&SeasonType=Regular%20Season&sort=PLAYER_NAME&dir=-1"
        driver = webdriver.Firefox()
        driver.get(pas_url)
        element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "I Accept")]')))
        element.click()
        select = Select(driver.find_element_by_xpath(
            r"/html/body/main/div/div/div[2]/div/div/nba-stat-table/div[1]/div/div/select"))
        select.select_by_index(0)
        src = driver.page_source
        soup = BeautifulSoup(src, "lxml")
        table = soup.find("div", attrs={"class": "nba-stat-table__overflow"})
        headers = table.findAll('th')
        headerlist = [h.text.strip() for h in headers]
        headerlist = [h.replace('\xa0', '') for h in headerlist]
        rows = table.findAll('tr')[1:]
        stats = [[td.getText().strip() for td in rows[i].findAll('td')] for i in range(len(rows))]
        passes = pd.DataFrame(stats, columns=headerlist)
        passes.drop(columns=passes.columns[2:7], inplace=True)
        self.passes = self.passes.append(passes[['Player', 'Team', 'PassesMade', 'PassesReceived',
                                                 'AST', 'SecondaryAST', 'AST', 'ASTToPass%']])
        del pas_url, passes

    # def collect_isolation(self, year):

