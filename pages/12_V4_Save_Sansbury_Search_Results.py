import streamlit as st
import os.path

try:
    # import scraper
    file = open('myfile.txt', 'w+')
    L = ["This is Delhi \n", "This is Paris \n", "This is London"]
    file.writelines(L)
    file.close()

    file1 = open("myfile.txt", "r")
    print("Output of Readlines after appending")
    st.info(file1.read())
    file1.close()

# except ImportError:
#     print("Oops! Cannot find the scraper.py file")
except Exception as e:
    st.info(e)
