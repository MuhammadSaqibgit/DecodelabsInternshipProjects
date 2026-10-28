# Decodelabs Internship - Week 1, Project 1
# Rule-Based AI Chatbot

# knowledge base: user message (key) -> bot reply (value)
responses = {
    # greetings
    "hello": "Hello! How can I help you today?",
    "hi": "Hi there! What would you like to talk about?",
    "hey": "Hey! Nice to meet you.",
    "good morning": "Good morning! Hope you have a great day.",
    "good night": "Good night! Take care.",

    # normal conversation
    "how are you": "I'm doing well, thanks for asking! How are you?",
    "what is your name": "I'm a ChatBot, a simple rule-based chatbot.",
    "who created you": "I was created by Muhammad Saqib.",
    "what can you do": "I can reply to a small set of messages that I was programmed with.",
    "tell me a joke": "Why did the computer go to the doctor? Because it had a virus!",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
    "help": "You can say hello, ask my name, ask how I am, ask for a joke, or type bye to exit.",

    # exit messages
    "bye": "Goodbye! It was nice chatting with you.",
    "goodbye": "Goodbye! It was nice chatting with you.",
    "exit": "Goodbye! Chat is ending now.",
    "quit": "Goodbye! Chat is ending now.",
}

exit_commands = ["bye", "goodbye", "exit", "quit"]

fallback = "Sorry, I don't understand that yet. Try something else or type 'help'."


print("=" * 45)
print("          Welcome to ChatBot!")
print("      A simple rule-based chatbot")
print("=" * 45)
print("Type 'help' to see what I can do.")
print("Type 'bye' to exit.\n")

while True:
    try:
        user_input = input("You: ")
    except (EOFError, KeyboardInterrupt):
        print("\nChatBot: Chat ended. Bye!")
        break

    # sanitize: remove extra spaces and make it lowercase
    cleaned_input = user_input.strip().lower()

    # empty message
    if cleaned_input == "":
        print("ChatBot: Please type something.")

    # exit commands
    elif cleaned_input in exit_commands:
        print("ChatBot:", responses[cleaned_input])
        break

    # special rule: "my name is ..."
    elif cleaned_input.startswith("my name is "):
        name = cleaned_input[len("my name is "):].title()
        print("ChatBot: Nice to meet you,", name + "!")

    # normal lookup in the dictionary, fallback if not found
    else:
        print("ChatBot:", responses.get(cleaned_input, fallback))
