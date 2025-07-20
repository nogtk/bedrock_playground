import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain.globals import set_debug
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

set_debug(True)

chat = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs={"max_tokens": 1000},
    region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
)

messages = [
    SystemMessage(content="あなたのタスクはユーザの質問に明確に答えることです。"),
    HumanMessage(content="空が青いのはなぜですか？"),
]

response = chat.invoke(messages)
print(response.content)
