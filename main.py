import os
import streamlit as st
from datetime import datetime
import pandas as pd
from langchain.chains import ConversationChain
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
from audio_recorder_streamlit import audio_recorder
from huggingface_hub import InferenceClient
from pydub import AudioSegment
import logging
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TherapistAssistant:
    def __init__(self):
        load_dotenv()
        self.setup_llm()
        self.setup_audio()
        self.load_session_state()
        
    def setup_llm(self):
        """Initialize LLM components"""
        try:
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key, 
                model_name="llama-3.3-70b-versatile"
            )
            self.hf_client = InferenceClient(
                api_key=os.getenv("HF_API_KEY")
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            st.error("Failed to initialize AI components. Please check your API keys.")

    def setup_audio(self):
        """Initialize audio components"""
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        
    def load_session_state(self):
        """Initialize or load session state"""
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        if "user_data" not in st.session_state:
            st.session_state["user_data"] = {
                "mood_history": [],
                "stress_history": []
            }

    def analyze_emotion(self, text: str) -> str:
        """Analyze emotion in text using HuggingFace model"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an emotion analyzer. Analyze the given text and determine if the person is feeling sad, happy, or neutral. Also provide a brief explanation of why you reached this conclusion."
                },
                {"role": "user", "content": text},
            ]
            
            completion = self.hf_client.chat_completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=messages,
                max_tokens=100,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return "Could not analyze emotion at this time."

    def process_audio(self, audio_bytes: bytes) -> Optional[str]:
        """Process recorded audio and return transcribed text"""
        temp_files = {"mp3": "temp_audio.mp3", "wav": "temp_audio.wav"}
        
        try:
            with open(temp_files["mp3"], "wb") as f:
                f.write(audio_bytes)
            
            audio = AudioSegment.from_mp3(temp_files["mp3"])
            audio.export(temp_files["wav"], format="wav")
            
            with sr.AudioFile(temp_files["wav"]) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                return text
                
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return None
            
        finally:
            for path in temp_files.values():
                if os.path.exists(path):
                    os.remove(path)

    def get_therapeutic_response(self, user_input: str) -> str:
        """Get AI response to user input"""
        prompt = ChatPromptTemplate.from_template(
            """You are an empathetic therapist assistant. Based on the following user input, provide:
            1. A brief emotional assessment
            2. Validation of their feelings
            3. One or two practical suggestions or coping strategies
            4. A gentle question to encourage self-reflection
            
            Keep the total response within 200 words and maintain a warm, supportive tone.
            
            User: {input}
            Assistant:"""
        )
        
        try:
            conversation = ConversationChain(llm=self.llm)
            return conversation.predict(input=user_input)
        except Exception as e:
            logger.error(f"Failed to get therapeutic response: {e}")
            return "I apologize, but I'm having trouble generating a response. Please try again."

    def save_progress(self, mood: int, stress_level: int):
        """Save user's mood and stress data"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "mood": mood,
            "stress_level": stress_level
        }
        st.session_state["user_data"]["mood_history"].append(entry)
        
        # Save to CSV
        try:
            df = pd.DataFrame(st.session_state["user_data"]["mood_history"])
            df.to_csv("mood_log.csv", index=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
            return False

    def render_ui(self):
        """Render the main UI"""
        st.title("🌟 AI Therapy Assistant")
        
        # Sidebar for progress tracking
        with st.sidebar:
            self.render_progress_tracking()
        
        # Main conversation interface
        st.subheader("💭 How are you feeling today?")
        
        # Text and audio input
        col1, col2 = st.columns([3, 1])
        with col1:
            user_input = st.text_input("Share your thoughts...", key="user_input")
        with col2:
            audio_bytes = audio_recorder()
            
        if audio_bytes:
            with st.spinner("Processing your message..."):
                text = self.process_audio(audio_bytes)
                if text:
                    st.info(f"I heard: {text}")
                    emotion = self.analyze_emotion(text)
                    st.write(f"Emotional Analysis: {emotion}")
        
        if st.button("Send", key="send_button"):
            if user_input:
                response = self.get_therapeutic_response(user_input)
                st.session_state["messages"].append({"user": user_input, "bot": response})
        
        # Display conversation history
        for msg in st.session_state["messages"]:
            st.text_area("You:", msg["user"], height=100, disabled=True)
            st.text_area("Assistant:", msg["bot"], height=150, disabled=True)
        
        # Additional features
        with st.expander("🎯 Daily Check-in"):
            self.render_daily_checkin()
        
        with st.expander("📚 Resources"):
            self.render_resources()
        
        # Emergency support
        if st.checkbox("🆘 I need immediate support"):
            st.error("""
                If you're in crisis, please reach out:
                - National Crisis Hotline: 988
                - Crisis Text Line: Text HOME to 741741
                - Emergency Services: 911
            """)

    def render_progress_tracking(self):
        """Render the progress tracking sidebar"""
        st.sidebar.header("📊 Your Progress")
        if st.session_state["user_data"]["mood_history"]:
            df = pd.DataFrame(st.session_state["user_data"]["mood_history"])
            st.sidebar.line_chart(df[["mood", "stress_level"]])

    def render_daily_checkin(self):
        """Render the daily check-in section"""
        mood = st.slider("Rate your mood (1-10):", 1, 10, 5)
        stress = st.slider("Rate your stress level (1-10):", 1, 10, 5)
        
        if st.button("Save Check-in"):
            if self.save_progress(mood, stress):
                st.success("Progress saved successfully!")
            else:
                st.error("Failed to save progress")

    def render_resources(self):
        """Render the resources section"""
        resource_type = st.selectbox(
            "What kind of resources are you looking for?",
            ["Meditation", "Anxiety Management", "Depression Support", "Sleep Hygiene"]
        )
        
        if resource_type:
            response = self.llm.predict(
                f"Provide 3 practical, evidence-based strategies for {resource_type} in bullet points."
            )
            st.write(response)

def main():
    assistant = TherapistAssistant()
    assistant.render_ui()

if __name__ == "__main__":
    main()