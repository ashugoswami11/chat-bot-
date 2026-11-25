import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.chat.util import Chat, reflections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import json
import re
import random
import string 


# Data Preprocessing
class TextPreprocessing:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))    
        
    def remove_punctuation(self, text):
        return text.translate(str.maketrans('','', string.punctuation))
    
    def tokenize(self, text):
        return word_tokenize(text)
    
    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stopwords]
    
    def preprocess(self, text):
        text = self.remove_punctuation(text)
        text = text.lower()
        tokens = self.tokenize(text)
        tokens = self.lemmatize(tokens)
        tokens = self.remove_stopwords(tokens)
        return " ".join(tokens)

class AiPoweredChatBot:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.pipeline = make_pipeline(self.vectorizer, self.model)
        self.train_data()
    
    def train_data(self):
        training_sentences = [
            "hello", "hi", "hey", "goodbye", "bye", "how are you", "I am fine", "thanks", "thank you",
            "what is your name", "who created you", "tell me a joke", "I love you"
        ]
        responses = [
            "Hi there!", "Hi there!", "Hello!", "Goodbye!", "Bye!", "I'm doing great, thanks!", "That's good to hear!",
            "you're welcome!","you're wecome!",
            "I'm a chatbot!","I was created by a developer!",
            "why don't scientists trust atoms? Because they make up everything!", "I love you too!" 
        ]
        
        self.pipeline.fit(training_sentences, responses)
        
    def chatbot_response(self, user_input):
        return self.pipeline.predict([user_input])[0] 
    
#basic rulebased chatbot
def chatbot_response(user_input):
    user_input = user_input.lower()
    if "hello" in user_input or "hi" in user_input:
        return "hello, how can i help you?"
    elif "how are you" in user_input:
        return "i am just a chatbot i don't have any feelings"
    elif "your name" in user_input:
        return "i am a chatbot what can i do for you?"
    elif "bye" in user_input:
        return "good-bye! have a great day"
    else:
        return "i am sorry, i don't understand your question"
    
print("Data Preprocessing Examples")
preprocessor = TextPreprocessing()
ai_bot = AiPoweredChatBot()
print("\n")
print("Basic Rule based chatbot")
print("chatbot:  Hello, How can i help you?")

pairs = [
    [
        r"hi|hello|hey",
        ["hello, How can I help you?", "Hey there! what's up?", "Hi! how's it going?"]
    ],
    [
        r"how are you?",
        ["i am a chatbot, but i'm doing great! How about you?", "i am chatbot. Your virtual assistant"]
    ],
    [
        r"(.*) your name?",
        ["My name is chatbot. What's yours?", "I am Chatbot, your virtual assistant."]
    ],
    [
        r".*help.*",
        ["I can help you with general queries. What do you need help with?", "Sure! Let me know your question"]
    ],
    [
        r"bye|goodbye",
        ["Goodbye! Have a great day!", "bye! Take care!"]
    ],
    [
        r"(.*)",
        ["I'm sorry, I don't understand your question.", "Please ask me a valid question."]
    ]
]

chatbot = Chat(pairs, reflections)
while True:
    user_input = input("user: ")
    preprocessed_text = preprocessor.preprocess(user_input)
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye! Have a great day!")
        break
    # response = chatbot.respond(user_input.lower())
    response = ai_bot.chatbot_response(preprocessed_text) 
    print("Chatbot: ", response)
     
     
# print(preprocessor.preprocess(sample_text))
# preprocessed_text = preprocessor.preprocess(sample_text)
    
    
    
    
    
        