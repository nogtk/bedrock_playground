import streamlit as st
import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

st.title("Bedrock チャット")

chat = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"max_tokens": 1000},
    streaming=True,
    region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt := st.chat_input("何でも聞いてください。"):
    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # チャット用メッセージリストを作成（セッション状態は後で更新）
    chat_messages = [
        SystemMessage(content="あなたのタスクはユーザの質問に明確に答えることです。"),
        *st.session_state.messages,
        HumanMessage(content=prompt)
    ]
    
    with st.chat_message("assistant"):
        response = st.write_stream(chat.stream(chat_messages))
    
    # セッション状態を更新
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.messages.append(AIMessage(content=response))
