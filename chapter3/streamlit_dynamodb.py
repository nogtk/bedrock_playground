import streamlit as st
import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

st.title("Bedrock チャット")

if "session_id" not in st.session_state:
    st.session_state.session_id = "session_id"

if "history" not in st.session_state:
    st.session_state.history = DynamoDBChatMessageHistory(
        table_name="BedrockChatSessionTable", session_id=st.session_state.session_id
    )

if "chain" not in st.session_state:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたのタスクはユーザの質問に明確に答えることです。"),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="human_message"),
        ]
    )

    chat = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={"max_tokens": 1000},
        streaming=True,
        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
    )

    chain = prompt | chat
    st.session_state.chain = chain

if (st.button("履歴クリア")):
    st.session_state.history.clear()

for message in st.session_state.history.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt := st.chat_input("何でも聞いてください。"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = st.write_stream(
            st.session_state.chain.stream(
                {
                    "messages": st.session_state.history.messages,
                    "human_message": [HumanMessage(content=prompt)],
                },
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
        )
    st.session_state.history.add_user_message(prompt)
    st.session_state.history.add_ai_message(response)
    