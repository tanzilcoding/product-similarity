import streamlit as st

try:
    # import scraper
    file = open('myfile.txt', 'w+')
# except ImportError:
#     print("Oops! Cannot find the scraper.py file")
except Exception as e:
    st.info(e)
