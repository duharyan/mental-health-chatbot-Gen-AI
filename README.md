# mental-health-chatbot-Gen-AI

## Overview
The **AI Therapy Assistant** is a compassionate virtual assistant designed to provide emotional support, track mental health progress, and offer coping strategies based on user input. It integrates **Groq's Llama 3.3-70B-Versatile LLM**, **Hugging Face models**, and **Google Speech Recognition** to analyze user emotions, provide therapeutic responses, and track stress/mood levels over time.

## Features
- **Text & Voice Input Support:** Users can type or speak their thoughts, and the AI will process their input.
- **Emotion Analysis:** Uses Hugging Face models to assess emotions from user messages.
- **AI-Powered Therapeutic Responses:** Provides personalized coping strategies and emotional validation.
- **Daily Mood & Stress Tracking:** Users can log their mood and stress levels for self-monitoring.
- **Progress Visualization:** Displays mood and stress trends in a graphical format.
- **Resource Recommendations:** Offers strategies for meditation, anxiety management, depression support, and sleep hygiene.
- **Emergency Support Information:** Displays crisis helplines when immediate support is needed.

## Installation
### Prerequisites
- API keys for Groq, Hugging Face, and optionally Google Cloud Speech-to-Text

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/therapist-assistant.git
   cd therapist-assistant
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up API keys:
   - Create a `.env` file and add your API keys:
     ```
     GROQ_API_KEY=your_groq_api_key
     HF_API_KEY=your_huggingface_api_key
     ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
- **Text Chat:** Enter your thoughts in the text box and click "Send."
- **Voice Input:** Record a message using the microphone and get a response.
- **Mood & Stress Tracking:** Use sliders to log your daily emotional state.
- **View Progress:** Monitor your mental health trends in the sidebar.
- **Get Resources:** Select a topic to receive evidence-based strategies.
- **Emergency Support:** Click the checkbox if you need urgent crisis helplines.

## Future Improvements
- Integrate real-time sentiment tracking
- Expand therapy resources with interactive exercises
- Implement chatbot memory for long-term interactions
- Support multilingual responses

## License
This project is open-source under the MIT License.

