from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from agents.base_agent import BaseAgent
from config import MEMORY_WINDOW, OLLAMA_MODEL


class ChatAgent(BaseAgent):

    def __init__(self):
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=MEMORY_WINDOW
        )

        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0
        )

    def handle(self, query):
        history = self.memory.load_memory_variables({})["chat_history"]

        messages = [
            SystemMessage(
                content="""
        You are a helpful AI assistant.

        Remember information shared during the current conversation.
        """
            )
        ]

        messages.extend(history)
        messages.append(HumanMessage(content=query))

        response = self.llm.invoke(messages)

        self.memory.save_context(
            {"input": query},
            {"output": response.content}
        )

        return response.content
