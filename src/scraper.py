import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import uuid

BASE_URL = "https://forum.cannabisanbauen.net"

def setupBrowser():
    options = webdriver.FirefoxOptions()
    options.add_argument('--headless')
    browser = webdriver.Firefox(options=options)

    return  browser

def get_topics(url):
    browser = setupBrowser()
    browser.get(url)

    html = browser.page_source
    soup = BeautifulSoup(html, 'html.parser')

    results = []

    for result in soup.find_all('div', class_="fps-result"):
        id = str(uuid.uuid1)
        topic_attributes = { id: {} }

        topic_status = result.find_all('div', class_="topic-statuses")
        topic_headline = result.find_all('span', class_="ember-view")[0].get_text()
        topic_content = result.find_all('span', class_="ember-view")[1].get_text()
        topic_link = result.find_all('a', class_="search-link")[0].get('href')
        
        topic_attributes[str(id)]['topic_status'] = "Dieses Thema hat eine Lösung" in str(topic_status)
        topic_attributes[str(id)]['topic_headline'] = topic_headline.replace('\n', '').replace('  ', '')
        topic_attributes[str(id)]['topic_content'] = topic_content.replace('\n', '').replace('  ', '')
        topic_attributes[str(id)]['topic_link'] = BASE_URL + topic_link.replace('\n', '').replace('  ', '')
        topic_attributes[str(id)]['topic_id'] = str(id)

        if topic_attributes[str(id)]['topic_status'] == True:
            results.append(topic_attributes[str(id)])

    browser.quit()

    return results


def scrape_url(query):
    url = BASE_URL + "/search?expanded=true&q=" + query.lower().replace(" ", "%20")
    print('query forum with search params. url: ' + url)

    results = get_topics(url)

    return results
