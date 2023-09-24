import os
import sys
import time
from langchain import LLMChain
import openai
import pinecone
import streamlit as st
from streamlit_chat import message
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
import traceback
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

try:
    import environment_variables
except ImportError:
    pass

try:
    # Setting page title and header
    st.set_page_config(
        page_title="Task 2: Using Generated Search Terms", page_icon=":robot_face:")
    st.markdown("<h1 style='text-align: center;'>Task 2: Using Generated Search Terms 😬</h1>",
                unsafe_allow_html=True)

    # Get environment variables
    # openai.organization = os.environ['openai_organization']
    # =======================================================
    OPENAI_API_KEY = os.environ['openai_api_key']
    pinecone_api_key = os.environ['pinecone_api_key_1']
    pinecone_environment = os.environ['pinecone_environment_1']
    openai.api_key = OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    index_name = os.environ['index_name_1']
    # ==================================================== #

    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_environment  # find next to API key in console
    )

    # connect to index
    index = pinecone.Index(index_name)
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

    # switch back to normal index for langchain
    index = pinecone.Index(index_name)

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
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
    if 'keyword_list' not in st.session_state:
        st.session_state['keyword_list'] = []
    if 'product_list' not in st.session_state:
        st.session_state['product_list'] = []

    # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
    st.sidebar.title("Sidebar")
    model_name = st.sidebar.radio("Choose a model:", ("GPT-4", "GPT-3.5"))
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

    def get_prompt():
        # Define the system message template
        system_template = """ 
        Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I do not know". Do not try to make up an answer. 
        
        #####Start of Context#####
        {context}
        #####End of Context#####
        """

        user_template = "Question:```{question}```"

        # Create the chat prompt templates
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)

        return prompt

    # generate a response
    def generate_response(prompt):
        query = prompt
        st.session_state['messages'].append(
            {"role": "user", "content": prompt})

        ######################################################
        # template = """

        # The following is a friendly conversation between a human and an AI.
        # The human wants to know keywords for highly specific search queries of grocery stores.
        # The answer format should be a comma separated list of suggested highly specific keywords that online shoppers use.
        # If you cannot answer the question, answer with "I do not know".

        # #####Start of Examples#####
        # Search term: Chicken
        # Keywords for highly specific search queries: chicken breasts, chicken legs

        # Search term: Eggs
        # Keywords for highly specific search queries: large eggs, medium eggs

        # Search term: Milk
        # Keywords for highly specific search queries: semi skimmed milk, whole milk

        # Search term: bread
        # Keywords for highly specific search queries: Medium Sliced White Bread, Medium Sliced Wholemeal Bread

        # Search term: Chicken breasts
        # Keywords for highly specific search queries: Fresh Chicken Breast Fillets, Fresh Chicken Breast Fillets

        # Search term: Lettuce
        # Keywords for highly specific search queries: Iceberg Lettuce, Little Gem Lettuce, Sweet Gem Lettuce

        # Search term: Tomatoes
        # Keywords for highly specific search queries: Round Tomatoes, Cherry Tomatoes, Baby Plum Tomatoes

        # Search term: Bananas
        # Keywords for highly specific search queries: Bananas Loose, Fairtrade Bananas

        # Search term: Apples
        # Keywords for highly specific search queries: Gala Apples, Braeburn Apples, Pink Lady Apples

        # Search term: Rice
        # Keywords for highly specific search queries: Basmati Rice, White Rice

        # Search term: Pasta
        # Keywords for highly specific search queries: Fusilli, Penne, Rigatoni
        # #####End of Examples#####

        # Current conversation:
        # Human: What are some keywords for highly specific search queries when the search is {input}?
        # AI Assistant:"""

        template = """

        You are a helpful assistant. You focus on the United Kingdom only. You are an expert in grocery items and shopping. 
        The human wants to know keywords for highly specific search queries of grocery stores. 
        The answer format should be a comma separated list of suggested highly specific keywords that online shoppers use. 
        If you cannot answer the question, answer with "I do not know".

        #####Start of Examples#####
        Search term: Chicken
        Keywords for highly specific search queries: chicken breasts, chicken legs

        Search term: Eggs
        Keywords for highly specific search queries: large eggs, medium eggs

        Search term: Milk
        Keywords for highly specific search queries: semi skimmed milk, whole milk

        Search term: bread
        Keywords for highly specific search queries: Medium Sliced White Bread, Medium Sliced Wholemeal Bread

        Search term: Chicken breasts
        Keywords for highly specific search queries: Fresh Chicken Breast Fillets, Fresh Chicken Breast Fillets

        Search term: Lettuce
        Keywords for highly specific search queries: Iceberg Lettuce, Little Gem Lettuce, Sweet Gem Lettuce

        Search term: Tomatoes
        Keywords for highly specific search queries: Round Tomatoes, Cherry Tomatoes, Baby Plum Tomatoes

        Search term: Bananas
        Keywords for highly specific search queries: Bananas Loose, Fairtrade Bananas

        Search term: Apples
        Keywords for highly specific search queries: Gala Apples, Braeburn Apples, Pink Lady Apples

        Search term: Rice
        Keywords for highly specific search queries: Basmati Rice, White Rice

        Search term: Pasta
        Keywords for highly specific search queries: Fusilli, Penne, Rigatoni
        #####End of Examples#####
        

        Current conversation:
        Human: What are some keywords for highly specific search queries when the search is {input}?
        AI Assistant:"""

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            template)

        example_human_history = HumanMessagePromptTemplate.from_template("Hi")
        example_ai_history = AIMessagePromptTemplate.from_template(
            "hello, how are you today?")

        human_template = "{input}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            human_template)

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, example_human_history, example_ai_history, human_message_prompt])

        chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        chain = LLMChain(llm=chat, prompt=chat_prompt)

        keyword_list = chain.run(query)

        if "Keywords for highly specific search queries:" in keyword_list:
            keyword_list = keyword_list.replace(
                "Keywords for highly specific search queries:", "")
        else:
            pass
        keyword_list = keyword_list.strip()

        # # initializing replace string
        # repl = "Keywords for highly specific search queries:"

        # # initializing substring to be replaced
        # subs = ""

        # # re.IGNORECASE ignoring cases
        # # compilation step to escape the word for all cases
        # compiled = re.compile(re.escape(subs), re.IGNORECASE)
        # keyword_list = compiled.sub(repl, keyword_list)
        # keyword_list = str(keyword_list)
        # keyword_list = keyword_list.strip()
        ######################################################

        ######################################################
        product_list = []
        if keyword_list == "I do not know":
            pass
        else:
            temp_keyword_list = keyword_list.split(",")

            for keyword in temp_keyword_list:
                keyword = keyword.strip()
                docs_and_scores = vectorstore.similarity_search_with_score(
                    keyword,
                )
                product_list.append(
                    {"keyword": keyword, "docs_and_scores": docs_and_scores})
        ######################################################

        return keyword_list, product_list

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            keyword_list, product_list = generate_response(
                user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append("")
            st.session_state['model_name'].append(model_name)
            st.session_state['keyword_list'].append(keyword_list)
            st.session_state['product_list'].append(product_list)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i],
                        is_user=True, key=str(i) + '_user')
                # message(st.session_state["generated"][i], key=str(i))
                keyword_list = st.session_state["keyword_list"][i]
                docs_and_scores = st.session_state["generated"][i]
                keyword_list = keyword_list.split(",")
                product_list = st.session_state["product_list"][i]

                st.markdown(
                    f"""<h2 style="word-wrap:break-word;">Suggested Keywords:</h2>""", unsafe_allow_html=True)

                counter = 0
                for keyword in keyword_list:
                    counter = counter + 1
                    keyword = keyword.strip()
                    st.markdown(
                        f"""<h3 style="word-wrap:break-word; color: violet;">{counter}. {keyword}</h3>""", unsafe_allow_html=True)

                    # st.markdown(f"""<span style="word-wrap:break-word;">{keyword_list}""", unsafe_allow_html=True)

                    for product in product_list:
                        if keyword == product["keyword"]:
                            docs_and_scores = product["docs_and_scores"]
                            counter = 0
                            for doc in docs_and_scores:
                                counter = counter + 1
                                score = list(doc)[1]
                                score = float(score)
                                score = score * 100
                                score = str(round(score, 2))
                                title = list(doc)[0].metadata['title']
                                price = list(doc)[0].metadata['price']
                                url = list(doc)[0].metadata['url']
                                image = list(doc)[0].metadata['image']
                                categories = list(
                                    doc)[0].metadata['categories']

                                st.markdown(
                                    f"""<h4 style="word-wrap:break-word;">Product {counter}:</h4>""", unsafe_allow_html=True)
                                st.markdown(
                                    f"""<span style="word-wrap:break-word;"><strong>Similarity score:</strong> {score}%""", unsafe_allow_html=True)
                                st.markdown(
                                    f"""<span style="word-wrap:break-word;"><a href="{url}" target="_blank">{title}</a>""", unsafe_allow_html=True)
                                st.markdown(
                                    f"""<span style="word-wrap:break-word;"><strong>Price:</strong> {price}""", unsafe_allow_html=True)
                                st.markdown(
                                    f"""<span style="word-wrap:break-word;"><strong>Categories:</strong> {categories}""", unsafe_allow_html=True)
                                st.markdown(
                                    f"""<span style="word-wrap:break-word;"><img src={image}>""", unsafe_allow_html=True)
                                # st.text(docs_and_scores)


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
    st.error('ERROR MESSAGE: {}'.format(error_message))
    st.error('ERROR MESSAGE: {}'.format(error_message), icon="🚨")
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    st.error(f'Error Type: {exc_type}', icon="🚨")
    st.error(f'File Name: {fname}', icon="🚨")
    st.error(f'Line Number: {exc_tb.tb_lineno}', icon="🚨")
    st.error(traceback.format_exc())
