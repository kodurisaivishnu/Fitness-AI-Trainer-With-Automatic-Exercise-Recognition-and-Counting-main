import os
import html
from dotenv import load_dotenv
import streamlit as st
from typing import Literal
from dataclasses import dataclass

load_dotenv()

@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

# Built-in fitness knowledge base for fallback when no OpenAI key
FITNESS_KB = {
    "warm up": "Always warm up for 5-10 minutes before exercising. Try light jogging, jumping jacks, or dynamic stretches to increase blood flow and prevent injuries.",
    "cool down": "Cool down after every workout with 5-10 minutes of static stretching. This helps reduce muscle soreness and improves flexibility.",
    "push up": "Push-ups target chest, shoulders, and triceps. Keep your body straight like a plank. Lower until your chest nearly touches the floor, then push back up. Start with 3 sets of 10 reps.",
    "squat": "Squats work your quads, hamstrings, and glutes. Stand with feet shoulder-width apart, lower your hips as if sitting in a chair, keep your knees behind your toes, and push back up. Aim for 3 sets of 15 reps.",
    "bicep curl": "Bicep curls isolate the biceps. Keep your elbows close to your body, curl the weight up slowly, and lower it with control. Don't swing your body. Try 3 sets of 12 reps.",
    "shoulder press": "Shoulder press targets your deltoids and triceps. Press weights overhead from shoulder height, fully extend your arms, then lower back down. Keep your core tight. Try 3 sets of 10 reps.",
    "weight loss": "For weight loss, combine cardio (20-30 min, 3-4 times/week) with strength training (2-3 times/week). Create a calorie deficit of 500 cal/day for ~1 lb loss per week. Focus on whole foods, lean protein, and vegetables.",
    "muscle gain": "For muscle gain, focus on progressive overload - gradually increase weight/reps. Eat in a slight calorie surplus with 1.6-2.2g protein per kg bodyweight. Rest 48 hours between training the same muscle group.",
    "stretching": "Stretch all major muscle groups after workouts. Hold each stretch for 15-30 seconds. Never bounce during stretches. Focus on hamstrings, quads, shoulders, and chest.",
    "rest": "Rest days are crucial for recovery and muscle growth. Take 1-2 rest days per week. Active recovery like walking or yoga is great on rest days.",
    "beginner": "If you're a beginner, start with bodyweight exercises 3 times per week. Focus on proper form before adding weight. A good starter routine: push-ups, squats, lunges, planks, and bicep curls.",
    "plank": "Planks strengthen your core. Keep your body straight from head to heels, engage your abs, and hold. Start with 20-30 seconds and build up to 60+ seconds.",
    "protein": "Aim for 1.6-2.2g of protein per kg of bodyweight daily for muscle growth. Good sources: chicken, fish, eggs, Greek yogurt, lentils, and protein shakes.",
    "calories": "Your daily calorie needs depend on age, gender, weight, and activity level. Use a TDEE calculator for an estimate. For maintenance, eat at your TDEE. For fat loss, eat 300-500 below.",
    "injury": "If you feel sharp pain during exercise, stop immediately. Rest, ice, compress, and elevate (RICE). Don't push through joint pain. Consult a doctor for persistent pain.",
    "motivation": "Stay motivated by setting specific goals, tracking progress, working out with a partner, varying your routine, and celebrating small wins. Consistency beats perfection!",
}

def _fallback_response(user_input):
    """Generate a response from built-in fitness knowledge when no API key is available."""
    user_lower = user_input.lower()

    # Check for keyword matches
    best_match = None
    best_score = 0
    for keyword, response in FITNESS_KB.items():
        words = keyword.split()
        score = sum(1 for w in words if w in user_lower)
        if score > best_score:
            best_score = score
            best_match = response

    if best_match and best_score > 0:
        return best_match

    # Generic helpful response
    if any(w in user_lower for w in ["hello", "hi", "hey"]):
        return "Hello! I'm your AI fitness assistant. Ask me about exercises, nutrition, workout plans, or form tips. What would you like to know?"

    if any(w in user_lower for w in ["thank", "thanks"]):
        return "You're welcome! Keep up the great work with your fitness journey. Feel free to ask anything else!"

    return ("Great question! As your AI fitness coach, I can help with exercise form, workout plans, nutrition basics, and injury prevention. "
            "Try asking about specific exercises (push-ups, squats, curls), weight loss, muscle gain, or stretching tips!")


def _get_openai_key():
    """Get OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return key if key and not key.startswith("your_") else None


def initialize_session_state():
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation" not in st.session_state:
        openai_key = _get_openai_key()
        if openai_key:
            try:
                from langchain_community.chat_models import ChatOpenAI
                from langchain.chains import ConversationChain
                from langchain.chains.conversation.memory import ConversationSummaryMemory

                llm = ChatOpenAI(
                    temperature=0,
                    openai_api_key=openai_key,
                    model_name="gpt-4o-mini"
                )
                conversation_memory = ConversationSummaryMemory(llm=llm)
                conversation_memory.save_context(
                    {"human": ""},
                    {"ai": "You are a chatbot inserted in a web app that uses AI to classify and count the repetitions of home exercises. Act as an expert in fitness and respond to the user as their personal AI trainer."}
                )
                st.session_state.conversation = ConversationChain(
                    llm=llm,
                    memory=conversation_memory,
                )
            except Exception as e:
                st.session_state.conversation = None
                st.warning(f"Could not initialize AI chatbot: {e}. Using built-in fitness knowledge instead.")
        else:
            st.session_state.conversation = None


def on_click_callback():
    human_prompt = st.session_state.get('human_prompt', '')
    if human_prompt:
        # Get response from LLM or fallback
        if st.session_state.get("conversation"):
            try:
                llm_response = st.session_state.conversation.run(human_prompt)
            except Exception as e:
                llm_response = _fallback_response(human_prompt)
        else:
            llm_response = _fallback_response(human_prompt)

        st.session_state.history.append(Message("human", human_prompt))
        st.session_state.history.append(Message("ai", llm_response))
        st.session_state.token_count += len(llm_response.split())
        st.session_state.human_prompt = ""


def chat_ui():
    initialize_session_state()
    st.title("Ask me anything about Fitness")

    # Show mode indicator
    if st.session_state.get("conversation"):
        st.caption("Powered by GPT - Ask any fitness question!")
    else:
        st.caption("Using built-in fitness knowledge (add OPENAI_API_KEY in .env for AI-powered responses)")

    chat_placeholder = st.container()
    prompt_placeholder = st.form("chat-form")

    with chat_placeholder:
        for chat in st.session_state.history:
            # Sanitize message to prevent XSS
            safe_message = html.escape(chat.message)
            if chat.origin == "human":
                st.markdown(f"**You:** {safe_message}")
            else:
                st.markdown(f"**Coach:** {safe_message}")

    with prompt_placeholder:
        st.text_input("Chat", key="human_prompt", placeholder="Ask about exercises, form, nutrition...")
        st.form_submit_button("Submit", on_click=on_click_callback)


if __name__ == "__main__":
    chat_ui()
