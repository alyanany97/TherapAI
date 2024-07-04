import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import json
import random
import logging
from datetime import datetime
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
from ttkbootstrap import Style, ttk

nltk.download('punkt')
nltk.download('wordnet')

class MentalHealthChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            with open('intents.json', 'r', encoding='utf-8') as file:
                self.intents = json.load(file)
        except FileNotFoundError:
            print("Error: intents.json file not found. Please make sure it exists in the same directory as the script.")
            self.intents = {"intents": []}
        except json.JSONDecodeError:
            print("Error: intents.json is not a valid JSON file. Please check its content.")
            self.intents = {"intents": []}
        self.conversation_log = []
        self.setup_logging()
        self.setup_nlp_pipeline()

    def setup_logging(self):
        logging.basicConfig(filename='chatbot.log', level=logging.INFO,
                            format='%(asctime)s:%(levelname)s:%(message)s')

    def setup_nlp_pipeline(self):
        # Load pre-trained model and tokenizer
        model_name = "microsoft/DialoGPT-medium"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set up sentiment analysis pipeline
        self.sentiment_pipeline = pipeline("sentiment-analysis")

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        return " ".join([self.lemmatizer.lemmatize(word) for word in tokens])

    def get_response(self, user_input):
        # Encode the user input
        input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
        
        # Generate a response
        chat_history_ids = self.model.generate(
            input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )
        
        # Decode the response
        response = self.tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        # If the AI response is not relevant, try to find a relevant intent-based response
        if not self.is_relevant_to_mental_health(response):
            intent_response = self.get_intent_based_response(user_input)
            if intent_response != "I'm not sure how to respond to that. Can you please rephrase or ask me something else about mental health?":
                return intent_response
        
        # If we couldn't find a relevant intent-based response, use the AI response anyway
        return response

    def is_relevant_to_mental_health(self, response):
        mental_health_keywords = ['anxiety', 'depression', 'stress', 'therapy', 'counseling', 'mental health', 
                                  'emotion', 'feeling', 'mood', 'well-being', 'self-care', 'mindfulness', 
                                  'relaxation', 'coping', 'support', 'help', 'listen', 'understand', 'experience']
        return any(keyword in response.lower() for keyword in mental_health_keywords)

    def get_intent_based_response(self, user_input):
        user_input = user_input.lower()
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                if pattern.lower() in user_input:
                    return random.choice(intent['responses'])
        return "I'm not sure how to respond to that. Can you please rephrase or ask me something else about mental health?"

    def sentiment_analysis(self, text):
        result = self.sentiment_pipeline(text)[0]
        return result['label']

    def log_conversation(self, user_input, bot_response, sentiment):
        self.conversation_log.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'user_input': user_input,
            'bot_response': bot_response,
            'sentiment': sentiment
        })
        logging.info(f"User: {user_input} | Bot: {bot_response} | Sentiment: {sentiment}")

    def save_conversation(self):
        df = pd.DataFrame(self.conversation_log)
        df.to_csv('conversation_log.csv', index=False)
        print("Conversation log saved to conversation_log.csv")

def get_response(self, user_input):
    # Encode the user input
    input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
    
    # Generate a response
    chat_history_ids = self.model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=self.tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )
    
    # Decode the response
    response = self.tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # If the AI response is not relevant, try to find a relevant intent-based response
    if not self.is_relevant_to_mental_health(response):
        intent_response = self.get_intent_based_response(user_input)
        if intent_response != "I'm not sure how to respond to that. Can you please rephrase or ask me something else about mental health?":
            return intent_response
    
    # If we couldn't find a relevant intent-based response, use the AI response anyway
    return response

    def is_relevant_to_mental_health(self, response):
        mental_health_keywords = ['anxiety', 'depression', 'stress', 'therapy', 'counseling', 'mental health', 
                              'emotion', 'feeling', 'mood', 'well-being', 'self-care', 'mindfulness', 
                              'relaxation', 'coping', 'support', 'help', 'listen', 'understand', 'experience']
        return any(keyword in response.lower() for keyword in mental_health_keywords)

def get_intent_based_response(self, user_input):
    user_input = user_input.lower()
    for intent in self.intents['intents']:
        for pattern in intent['patterns']:
            if pattern.lower() in user_input:
                return random.choice(intent['responses'])
    return "I'm not sure how to respond to that. Can you please rephrase or ask me something else about mental health?"

    def sentiment_analysis(self, text):
        result = self.sentiment_pipeline(text)[0]
        return result['label']

    def log_conversation(self, user_input, bot_response, sentiment):
        self.conversation_log.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'user_input': user_input,
            'bot_response': bot_response,
            'sentiment': sentiment
        })
        logging.info(f"User: {user_input} | Bot: {bot_response} | Sentiment: {sentiment}")

    def save_conversation(self):
        df = pd.DataFrame(self.conversation_log)
        df.to_csv('conversation_log.csv', index=False)
        print("Conversation log saved to conversation_log.csv")

class ChatbotGUI:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.style = Style(theme="flatly")
        self.window = self.style.master
        self.window.title("Mental Health Chatbot")
        self.window.geometry("600x700")
        self.setup_ui()

    def setup_ui(self):
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, width=60, height=25, font=("Helvetica", 10)
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_display.config(state=tk.DISABLED)

        # Input area
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        input_frame.grid_columnconfigure(0, weight=1)

        self.input_field = ttk.Entry(input_frame, font=("Helvetica", 10))
        self.input_field.grid(row=0, column=0, sticky="ew")

        self.send_button = ttk.Button(
            input_frame, text="Send", command=self.send_message, style="Accent.TButton"
        )
        self.send_button.grid(row=0, column=1, padx=(10, 0))

        # Bind the Return key to send_message
        self.window.bind('<Return>', lambda event: self.send_message())

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to chat")
        self.status_bar = ttk.Label(
            main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 0))

        self.display_message("Chatbot: Hello! I'm an AI assistant here to discuss mental health. How can I support you today?")

    def send_message(self):
        user_input = self.input_field.get()
        if user_input.lower() == 'exit':
            self.chatbot.save_conversation()
            self.window.quit()
        else:
            self.display_message(f"You: {user_input}")
            self.input_field.delete(0, tk.END)
            self.status_var.set("Processing...")
            threading.Thread(target=self.get_bot_response, args=(user_input,), daemon=True).start()

    def get_bot_response(self, user_input):
        response = self.chatbot.get_response(user_input)
        sentiment = self.chatbot.sentiment_analysis(user_input)
        self.chatbot.log_conversation(user_input, response, sentiment)
        self.display_message(f"Chatbot: {response}")
        self.status_var.set("Ready to chat")

    def display_message(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, message + "\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    chatbot = MentalHealthChatbot()
    gui = ChatbotGUI(chatbot)
    gui.run()