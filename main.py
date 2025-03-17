import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain, learn
from datetime import datetime


st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #28a745;
            color: white;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 15px;
            border: none;
            transition: 0.3s;
        }

    </style>
    """,
    unsafe_allow_html=True
)


st.title("CuraMate 🩺")
st.subheader("Doc's Favorite Assistant. ✅")
#Buttn to create the knowledge base
btn = st.button("Create Knowledgebase")

if btn:
    current_time = datetime.now()
    create_vector_db()
    done_time = datetime.now()
    st.write(f"✅ Knowledgebase Created...    [{((done_time - current_time).total_seconds())/60}]")
    
st.divider()
#Input field forthe question
question = st.text_input("Enter Your Medical Query:")

#Input field for new response (Only used when submitting a response)
response_input = st.text_area("Submit a Better Response:", height=100)

submit_response = st.button("Submit Response")
st.divider()
#If !!! the submit button is clicked, call learn() function from backabone AI
if submit_response and question and response_input:
    response_input = '"Recommended treatment : '+response_input+'"'
    print('\n\n\n\n\n'+response_input+'\n\n\n\n')
    learn(question, response_input.replace("\n", " "))
    
    st.success("✅ Your response has been added to the knowledge base! will take (2-3)mins⌚")



#basic question is entered, generate an answer
if question and not submit_response:  # Avoid running when submitting a response
    chain = get_qa_chain()
    response = chain(question)
    res = response["result"].split("\n")
    print(f'\n\n\n {res}\n\n\n\n')

    with st.expander("CuraMate has made the REPORT 🩺",expanded = True):
        st.markdown(response["result"])
    # for element in res :
    #     st.write(element)
