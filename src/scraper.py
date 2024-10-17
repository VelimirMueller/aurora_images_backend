import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import uuid
import os

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
        topic_image_sources = []

        print("fetching image sources from topic url: " + BASE_URL + topic_link)
        browser.get(BASE_URL + topic_link)
        html = browser.page_source
        soup = BeautifulSoup(html, 'html.parser')

        for image_wrapper in soup.find_all('div', class_="lightbox-wrapper"):
            print(image_wrapper.find('img').get('src'))
            topic_image_sources.append(image_wrapper.find('img').get('src'))
            
        print()
        print("found following relevant images: ")
        print(topic_image_sources)
        print()
        topic_attributes[str(id)]['topic_status'] = "Dieses Thema hat eine Lösung" in str(topic_status)
        topic_attributes[str(id)]['topic_headline'] = topic_headline.replace('\n', '').replace('  ', '')
        topic_attributes[str(id)]['topic_content'] = topic_content.replace('\n', '').replace('  ', '')
        topic_attributes[str(id)]['topic_link'] = BASE_URL + topic_link.replace('\n', '').replace('  ', '')
        topic_attributes[str(id)]['topic_images'] = topic_image_sources
        topic_attributes[str(id)]['topic_id'] = str(id)

        results.append(topic_attributes[str(id)])
        if len(results) > 1:
            break
    browser.quit()


    print()
    print("printing results:")
    #print(results)
    print()
    return results

# Function to download an image
def download_image(url, folder):
    response = requests.get(url)
    if response.status_code == 200:
        # Extract the image name from the URL
        image_name = url.split("/")[-1]
        # Create the folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Save the image to the specified folder
        with open(os.path.join(folder, image_name), 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {image_name}")
    else:
        print(f"Failed to retrieve image from {url}")

def scrape_url(query):
    url = BASE_URL + "/search?expanded=true&q=" + query.lower().replace(" ", "%20")
    print('query forum with search params. url: ' + url)

    results = get_topics(url)
    imageLinks = []

    for result in results:
        for link in result['topic_images']:
           imageLinks.append(link)

    for link in imageLinks:
        download_image(link, 'cache')
        

    return results
