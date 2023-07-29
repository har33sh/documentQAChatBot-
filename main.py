from module_rag.chatbot import ChatBot

bot = ChatBot()

if __name__ == '__main__':
    print("Please enter your query")
    while True:
        question = input()
        answer = bot.chat(question)
        print(answer)
