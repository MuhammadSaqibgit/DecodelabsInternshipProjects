# Rule-Based AI Chatbot

Decodelabs Internship - Week 1, Project 1

A simple command-line chatbot written in Python. It replies to predefined messages using rules (a dictionary and if-elif conditions). It does not use machine learning.

## Project 1 requirements and how they are covered

| Requirement | How it is done |
|---|---|
| Input loop | A `while True` loop keeps the chat running |
| Sanitization | `.strip().lower()` handles extra spaces and capital letters |
| Knowledge base | A dictionary with 17 predefined messages |
| Fallback response | `responses.get()` returns a default reply for unknown input |
| Exit strategy | `bye`, `goodbye`, `exit` or `quit` stops the loop with `break` |
| If-else logic | `if / elif / else` handles empty input, exit commands, "my name is ..." and normal replies |

## Files

- `main.py` - the chatbot
- `README.md` - this file

## How to run

You need Python 3. No extra packages are needed.

```
python main.py
```

On some Windows computers use `py main.py` instead.

## Example

```
You:    HELLO
ChatBot: Hello! How can I help you today?
You: my name is ali
ChatBot: Nice to meet you, Ali!
You: what is the weather on Mars
ChatBot: Sorry, I don't understand that yet. Try something else or type 'help'.
You: bye
ChatBot: Goodbye! It was nice chatting with you.
```

## Things you can try

- `hello`, `how are you`, `what is your name`, `tell me a joke`, `help`
- Capital letters and extra spaces, like `   HeLLo   `
- Pressing Enter without typing anything
- A random sentence, to see the fallback reply
- `bye` to exit

## Note

The bot only understands messages that exactly match its dictionary (after cleaning). Anything else gets the fallback reply. Adding more intents is just a matter of adding more entries to `responses`.
