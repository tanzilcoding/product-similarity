import os
import sys
import time
import json
import openai
import pinecone
import traceback
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_chat import message
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import LLMChain

try:
    import environment_variables
except ImportError:
    pass

try:
    # Setting page title and header
    st.set_page_config(page_title="Multiple keyword search at a time",
                       page_icon=":robot_face:")
    st.markdown("<h1 style='text-align: center;'>Multiple keyword search at a time</h1>",
                unsafe_allow_html=True)

    # Get environment variables
    # openai.organization = os.environ['openai_organization']
    # =======================================================
    OPENAI_API_KEY = os.environ['openai_api_key']
    openai.api_key = OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    # ==================================================== #
    pinecone_api_key_1 = os.environ['pinecone_api_key_1']
    pinecone_environment_1 = os.environ['pinecone_environment_1']
    index_name_1 = os.environ['index_name_1']
    # ==================================================== #
    pinecone_api_key_2 = os.environ['pinecone_api_key_2']
    pinecone_environment_2 = os.environ['pinecone_environment_2']
    index_name_2 = os.environ['index_name_2']
    # ==================================================== #

    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=pinecone_api_key_1,
        environment=pinecone_environment_1  # find next to API key in console
    )

    # connect to index
    index_1 = pinecone.Index(index_name_1)
    # wait a moment for the index to be fully initialized
    time.sleep(1)
    # stats = index.describe_index_stats()

    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=pinecone_api_key_2,
        environment=pinecone_environment_2  # find next to API key in console
    )

    # connect to index
    index_2 = pinecone.Index(index_name_2)
    # wait a moment for the index to be fully initialized
    time.sleep(1)
    # stats = index.describe_index_stats()

    # get openai api key from platform.openai.com
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'

    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )

    text_field = "text"

    vectorstore_1 = Pinecone(
        index_1, embed.embed_query, text_field
    )

    vectorstore_2 = Pinecone(
        index_1, embed.embed_query, text_field
    )

    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = []

    # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
    st.sidebar.title("Sidebar")
    model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    counter_placeholder = st.sidebar.empty()
    # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    clear_button = st.sidebar.button("Clear Conversation", key="clear")

    # Map model names to OpenAI model IDs
    if model_name == "GPT-3.5":
        model = "gpt-3.5-turbo-16k"
    else:
        model = "gpt-4"

    # Initialize the large language model
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        model_name=model,
    )

    # reset everything
    if clear_button:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.session_state['model_name'] = []
        # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

    def get_title_list(docs_and_scores):
        title_list = []

        for doc in docs_and_scores:
            title = list(doc)[0].metadata['title']
            title = title.strip()

            if len(title) > 0:
                title_list.append(title)

        return title_list

    def get_search_data(docs_and_scores):
        search_data = []

        for doc in docs_and_scores:
            product_data = {}
            document = list(doc)[0]
            metadata = document.metadata
            source = metadata['source']
            score = list(doc)[1]
            score = float(score)
            score = score * 100
            score = round(score, 2)
            title = list(doc)[0].metadata['title']
            price = list(doc)[0].metadata['price']
            url = list(doc)[0].metadata['url']
            image = list(doc)[0].metadata['image']
            categories = list(
                doc)[0].metadata['categories']

            product_data.update({"score": score})
            product_data.update({"title": title})
            product_data.update({"price": price})
            product_data.update({"url": url})
            product_data.update({"image": image})
            product_data.update({"categories": categories})
            product_data.update({"source": source})

            search_data.append(product_data)

        return search_data

    def get_similar_product_list(prompt):
        prompt = str(prompt)
        # query = prompt
        search_data = []
        print(f'prompt: {prompt}')
        tokens = prompt.split(" ")
        word_list = []
        st.session_state['messages'].append(
            {"role": "user", "content": prompt})

        for token in tokens:
            token = token.strip()
            if len(token) > 0:
                word_list.append(token)

        word_list = list(set(word_list))

        ######################################################
        query = ' '.join(word_list)

        if len(word_list) < 2:
            # template = """
            # My search term is delimited by triple backticks:
            # ```{query}```

            # I want to find out the most relevant keywords that I can use on online grocery stores.
            # I want two word keywords. Give me the top five most relevant keywords related to my search term.
            # Separate the keywords with comma. Give only the comma separated keywords and nothing else.
            # """

            template = """
            My search term is delimited by triple backticks: 
            ```{query}```
            
            I want to find out the most relevant keywords that I can use on online grocery stores. 
            I want three word keywords. Give me the top three most relevant keywords related to my search term. 
            Separate the keywords with comma. Give only the comma separated keywords and nothing else.

            Grocery keywords related to Eggs:
            - Free Range Eggs
            - Range Eggs Large
            - Range Eggs Medium

            Grocery keywords related to Milk:
            - Semi Skimmed Milk
            - British Skimmed Milk
            - British Whole Milk

            Grocery keywords related to bread:
            - Sliced White Bread
            - Medium White Bread
            - Sliced Wholemeal Bread

            Grocery keywords related to Chicken:
            - fresh whole chicken
            - fresh chicken legs
            - fresh chicken legs breasts
            - fresh skinless chicken

            Grocery keywords related to Chicken breasts:
            - Fresh Chicken Breast Fillets
            - Fresh Chicken Breast Skinless & Boneless
            - Fresh Chicken Breast Boneless
            - Fresh Diced Chicken Breast

            Grocery keywords related to Lettuce:
            - Iceberg Lettuce
            - Little Gem Lettuce
            - Sweet Gem Lettuce

            Grocery keywords related to Tomatoes:
            - Classic Round Tomatoes
            - Cherry Tomatoes
            - Baby Plum Tomatoes

            Grocery keywords related to Bananas:
            - Bananas Loose
            - Fairtrade Bananas

            Grocery keywords related to Apples:
            - Royal Gala Apples
            - Braeburn Apples
            - Pink Lady Apples

            Grocery keywords related to Rice:
            - Basmati Rice 1kg
            - White Rice 1kg
            - Long Grain Rice
            - Long White Rice

            Grocery keywords related to Pasta:
            - Fusilli 1kg
            - Penne 1kg
            - Rigatoni 1kg
            """

            system_message_prompt = SystemMessagePromptTemplate.from_template(
                template)

            human_template = "{query}"
            human_message_prompt = HumanMessagePromptTemplate.from_template(
                human_template)

            chat_prompt = ChatPromptTemplate.from_messages(
                [system_message_prompt, human_message_prompt])

            chain = LLMChain(llm=llm, prompt=chat_prompt)

            response = chain.run(
                {"query": query})

            keyword_list = response.split(",")

            # print("========================================================")
            for keyword in keyword_list:
                keyword = keyword.strip()
                # print(f'keyword: {keyword}')
                docs_and_scores = vectorstore_2.similarity_search_with_score(
                    keyword,
                )

                search_data = search_data + get_search_data(docs_and_scores)

            target_column = "score"
            is_reverse_sort = True
            temp_search_data = sorted(
                search_data, key=lambda d: d[target_column.lower()], reverse=is_reverse_sort)
            search_data = []
            title_list = []
            for temp_search in temp_search_data:
                title = temp_search["title"]
                title = title.strip()
                title = title.lower()
                if title not in title_list:
                    title_list.append(title)
                    search_data.append(temp_search)

                if len(search_data) >= 4:
                    break

        # elif len(word_list) == 2:
        else:
            # Step 1: We shall look into the product titles for al least 3 products
            docs_and_scores = vectorstore_2.similarity_search_with_score(
                query,
            )

            # search_data = get_search_data(docs_and_scores)
            title_list = get_title_list(docs_and_scores)

            # Step 2: Oops! Not even a product was found when we searched based on product titles?
            # Okay! Let us look into the full page data this time
            if len(title_list) < 1:
                docs_and_scores = vectorstore_1.similarity_search_with_score(
                    query,
                )

                title_list = get_title_list(docs_and_scores)

            if len(title_list) > 0:
                product_title_list = ""
                for title in title_list:
                    product_title_list = product_title_list + title + "\n\n"
                product_title_list = product_title_list.strip()

                template = """
                I am giving you a list of product titles. 
                Which product title is the most relevant to the term "{query}"? 
                I need a short answer. Mention only the product title. 

                =====================START OF PRODUCT TITLES=====================
                {product_title_list}
                =====================END OF PRODUCT TITLES=====================
                """

                system_message_prompt = SystemMessagePromptTemplate.from_template(
                    template)

                human_template = "{query}"
                human_message_prompt = HumanMessagePromptTemplate.from_template(
                    human_template)

                chat_prompt = ChatPromptTemplate.from_messages(
                    [system_message_prompt, human_message_prompt])

                chain = LLMChain(llm=llm, prompt=chat_prompt)

                most_relevant_product_title = chain.run(
                    {"query": query, "product_title_list": product_title_list})

                docs_and_scores = vectorstore_1.similarity_search_with_score(
                    most_relevant_product_title,
                )

                search_data = get_search_data(docs_and_scores)
        ######################################################

        return search_data

    def get_similar_products_by_title(prompt):
        query = prompt
        st.session_state['messages'].append(
            {"role": "user", "content": prompt})

        ######################################################
        # Step 1: We shall look into the product titles for al least 3 products
        docs_and_scores = vectorstore_2.similarity_search_with_score(
            query,
        )

        minimum_target_score = 100
        while minimum_target_score >= 70:
            search_data = []

            for doc in docs_and_scores:
                product_data = {}
                document = list(doc)[0]
                metadata = document.metadata
                source = metadata['source']
                score = list(doc)[1]
                score = float(score)
                score = score * 100
                score = round(score, 2)
                title = list(doc)[0].metadata['title']
                price = list(doc)[0].metadata['price']
                url = list(doc)[0].metadata['url']
                image = list(doc)[0].metadata['image']
                categories = list(
                    doc)[0].metadata['categories']

                product_data.update({"score": score})
                product_data.update({"title": title})
                product_data.update({"price": price})
                product_data.update({"url": url})
                product_data.update({"image": image})
                product_data.update({"categories": categories})
                product_data.update({"source": source})

                if score >= minimum_target_score:
                    search_data.append(product_data)

            minimum_target_score = minimum_target_score - 2

            if len(search_data) >= 3:
                break

        # Step 2: Oops! Not even a product was found when we searched based on product titles?
        # Okay! Let us look into the full page data this time
        if len(search_data) < 1:
            docs_and_scores = vectorstore_1.similarity_search_with_score(
                query,
            )

            minimum_target_score = 100
            while minimum_target_score >= 70:
                search_data = []

                for doc in docs_and_scores:
                    product_data = {}
                    document = list(doc)[0]
                    metadata = document.metadata
                    source = metadata['source']
                    score = list(doc)[1]
                    score = float(score)
                    score = score * 100
                    score = round(score, 2)
                    title = list(doc)[0].metadata['title']
                    price = list(doc)[0].metadata['price']
                    url = list(doc)[0].metadata['url']
                    image = list(doc)[0].metadata['image']
                    categories = list(
                        doc)[0].metadata['categories']

                    product_data.update({"score": score})
                    product_data.update({"title": title})
                    product_data.update({"price": price})
                    product_data.update({"url": url})
                    product_data.update({"image": image})
                    product_data.update({"categories": categories})
                    product_data.update({"source": source})

                    if score >= minimum_target_score:
                        search_data.append(product_data)

                minimum_target_score = minimum_target_score - 2

                if len(search_data) >= 3:
                    break
        ######################################################

        return search_data

    def generate_response(prompt):
        query = prompt
        search_data = []
        st.session_state['messages'].append(
            {"role": "user", "content": prompt})

        ######################################################
        tokens = query.split("\n")
        search_term_list = []
        for search_term in tokens:
            search_term = search_term.strip()
            if search_term == "":
                pass
            else:
                search_term_list.append(search_term)

        result_data = []
        for search_term in search_term_list:
            result_data = get_similar_product_list(search_term)
            ######################################################

            for result in result_data:
                # source = result['source']
                score = str(result['score'])
                title = result['title']
                # search_data.append({"Search Term": search_term, "Keyword": keyword,
                #                     "Similarity Score": score, "Product Title": title, "Similar Products": "", },)
                # search_data.append({"Search Term": search_term, "Similarity Score": score,
                #                    "Product Title": title, "Similar Products": "", },)
                similar_product_list = get_similar_products_by_title(title)
                counter = 0
                for similar_product_data in similar_product_list:
                    counter = counter + 1
                    similar_product_title = similar_product_data["title"]
                    similar_product_score = str(similar_product_data["score"])
                    similar_product_title = f'({similar_product_score}) {similar_product_title}'

                    if counter <= 1:
                        search_data.append({"Search Term": search_term, "Similarity Score": score,
                                           "Product Title": title, "Similar Products": similar_product_title, },)
                    else:
                        search_data.append({"Search Term": "", "Similarity Score": "",
                                           "Product Title": "", "Similar Products": similar_product_title, },)
        ######################################################

        return search_data

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = generate_response(
                user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['model_name'].append(model_name)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i],
                        is_user=True, key=str(i) + '_user')
                # message(st.session_state["generated"][i], key=str(i))
                search_data = st.session_state["generated"][i]

                df = pd.DataFrame(search_data)

                st.dataframe(df, use_container_width=True)

                st.markdown(
                    f"""<a href="https://jsongrid.com/json-grid" target="_blank">Convert JSON to HTML Table</a>""", unsafe_allow_html=True)

                st.info(json.dumps(search_data))

except Exception as e:
    error_message = ''
    # st.text('Hello World')
    st.error('An error has occurred. Please try again.', icon="🚨")
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
    print(traceback.format_exc())
