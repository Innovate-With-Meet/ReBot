import os
from dotenv import load_dotenv
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Setup LLM
llm = ChatOpenAI(temperature=0.7)

# Setup memory
memory = ConversationBufferMemory()

# Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Define chatbot response function
def chat(user_input, chat_history):
    response = conversation.predict(input=user_input)
    chat_history.append((user_input, response))
    return "", chat_history

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 LangChain Hackathon Chatbot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask me anything...")
    clear = gr.Button("Clear")

    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch app
if __name__ == "__main__":
    demo.launch()
