import chainlit as cl
from LLMRAGModel import model
import torch
import gc
from dotenv import load_dotenv
load_dotenv()
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.on_chat_start
def query_llm():
    llm_chain,foo = model.getnewChain()
    cl.user_session.set("llm_chain", llm_chain)
    print(f"Current user is: {cl.user_session.get('llm_chain')}")

@cl.on_message
async def query_llm(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    response=llm_chain.invoke(message.content)
    await cl.Message(response["text"]).send()
    torch.cuda.empty_cache()
    gc.collect()

