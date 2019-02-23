from selenium import webdriver
from selenium.webdriver.common.by import By
import argparse
import time, json

parser = argparse.ArgumentParser(description='crawler to get news and generate a dataset from it.')
parser.add_argument('--out', type=str, default= 'datasetEconomyNews.json', help='name of out file')
jsonOutName = parser.parse_args().out

def crawling():
    browser = webdriver.Chrome()
    browser.get('https://www.nytimes.com/section/business/economy')

    #aguarda 10 segundos
    for i in range(0, 29):
        time.sleep(3)
        browser.find_element_by_xpath('//*[@id="stream-panel"]/div[1]/div/div/div/button').click()

    time.sleep(5)

    headlineTitle = browser.find_elements_by_class_name("e1xfvim30")
    headlineText = browser.find_elements_by_class_name("e1xfvim31")
    
    dataset=[]

    for i in range(len(headlineText)):
        print("# "+headlineTitle[i].text)
        print("--> "+headlineText[i].text)
        print("\n")
        dataset.append(dict(idx=i,headlineTitle=headlineTitle[i].text, headlineText=headlineText[i].text, classification=[]))
    generateJson(dataset)

def generateJson(dataset):
    with open(jsonOutName, 'w') as datasetFile:
        datasetFile.write(json.dumps(dataset, sort_keys=True, indent=4, separators=(',', ': ')))

if __name__ == '__main__':
    crawling()

# https://www.nytimes.com/section/business/economy
# manchete: e1xfvim30
# sub-texto: e1xfvim31
