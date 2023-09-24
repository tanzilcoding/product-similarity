import streamlit as st


st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",
)

st.title("Main Page")
st.sidebar.success("Select a page above.")

st.text("Welcome!")

st.text("Please use the menu from the sidebar to access different part of this project.")
