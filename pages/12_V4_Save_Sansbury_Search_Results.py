import os
import re
import sys
import traceback
import csv
import time
import random
import streamlit as st
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from requests_html import HTMLSession
from io import StringIO
from html.parser import HTMLParser

# python scrape ajax page

# try:
#     import scraper
# except ImportError:
#     st.error('Sorry but the scraper.py file was not loaded.')

user_agents_list = [
    'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.83 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36'
]


def make_request(url, use_selenium=False):
    try:
        if use_selenium:
            page = driver.get(url)
            time.sleep(2)
            html = driver.page_source
        else:
            headers = {'User-Agent': random.choice(user_agents_list)}
            session = HTMLSession()
            response = session.get(url, headers=headers, verify=False)
            return response.text

        return html
    except Exception as e:
        print(e)
        return None


try:
    with st.form(key='Form1'):
        keywords = st.text_area(
            "Enter your keywords", )
        is_submitted = st.form_submit_button(label='Submit Keywords')

        if is_submitted:
            # driver_service = Service()
            # options = Options()
            # options.add_experimental_option("detach", True)
            # driver = webdriver.Chrome(service=driver_service, options=options)
            driver = webdriver.Chrome()

            csv_data = []
            # keyword_list = ["eggs", "milk", "lettuce"]
            keyword_list = keywords.split("\n")
            for keyword in keyword_list:
                keyword = keyword.strip()

                if len(keyword) > 0:
                    # print(keyword)
                    url = f'https://www.sainsburys.co.uk/gol-ui/SearchResults/{keyword}'

                    soup = ""
                    use_selenium = 'gol-ui' in url
                    page_html = make_request(url, use_selenium)
                    time.sleep(10)
                    page_html = make_request(url, use_selenium)
                    if page_html is None:
                        pass
                    else:
                        soup = BeautifulSoup(page_html, features='html.parser')
                        product_h2_titles = soup.find_all(
                            "h2", {"class": "pt__info__description"})

                        for product_h2_title in product_h2_titles:
                            product_title_link = product_h2_title.find(
                                'a', href=True)
                            product_title = product_title_link.text
                            csv_data.append([keyword, product_title])

                cwd = os.getcwd()
                csv_file = f'{cwd}/sainsbury-search-result-data.csv'

                if len(csv_data) > 0:
                    # create a file called test.csv
                    # and store it in a temporary variable
                    with open(csv_file, 'w+') as csv_file:
                        # pass the temp variable to csv.writer
                        # function
                        csv_writer = csv.writer(csv_file)

                        # pass the row values to be stored in
                        # different rows
                        csv_writer.writerows(csv_data)

except Exception as e:
    error_message = ''
    # st.text('Hello World')
    st.error('An error has occurred. Please try again.', icon="🚨")
    # print('An error has occurred. Please try again.', icon="🚨")
    # Just print(e) is cleaner and more likely what you want,
    # but if you insist on printing message specifically whenever possible...
    if hasattr(e, 'message'):
        error_message = e.message
    else:
        error_message = e
    st.error('ERROR MESSAGE: {}'.format(error_message), icon="🚨")
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    st.error(f'Error Type: {exc_type}', icon="🚨")
    st.error(f'File Name: {fname}', icon="🚨")
    st.error(f'Line Number: {exc_tb.tb_lineno}', icon="🚨")
    st.error(traceback.format_exc())

    # print('ERROR MESSAGE: {}'.format(error_message), icon="🚨")
    # exc_type, exc_obj, exc_tb = sys.exc_info()
    # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    # print(f'Error Type: {exc_type}', icon="🚨")
    # print(f'File Name: {fname}', icon="🚨")
    # print(f'Line Number: {exc_tb.tb_lineno}', icon="🚨")
    # print(traceback.format_exc())
