import os
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- CONFIGURATION & MODEL INITIALIZATION ---
# The API key is securely loaded from environment variables (e.g., Colab Secrets, Streamlit Secrets).
API_KEY = os.environ.get("GEMINI_API_KEY")

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        MODEL_NAME = "gemini-2.5-flash"
    except Exception as e:
        st.error(f"Failed to configure Gemini client: {e}")
        st.stop()
else:
    st.error("❌ GEMINI_API_KEY not found. Please set it as an environment variable (Secret) to run the app.")
    st.stop()


# ---------------------------
# CORE GENERATION LOGIC
# ---------------------------
def generate_resume_bullets(name, job_title, skills_list):
    """
    Calls the Gemini API to generate 5 high-impact resume bullet points.
    """
    
    # 1. Define the System Instruction (the LLM's role)
    # This highly restrictive prompt ensures the model acts as an expert writer.
    system_prompt = (
        "You are an expert career coach specializing in high-impact resume writing. "
        "Your task is to generate 5 powerful, job-specific resume bullet points. "
        "Each bullet MUST start with a strong action verb (past tense), quantify results, "
        "and follow the STAR method (Situation, Task, Action, Result) format where possible. "
        "The output MUST be a clean, unformatted list of exactly 5 bullet points, separated by newlines."
    )

    # 2. Define the User Prompt (the task and data)
    user_prompt = f"""
    Generate the 5 resume bullet points for the following profile:
    - Target Job Title: {job_title}
    - Key Skills to Emphasize: {', '.join(skills_list)}
    - Applicant Name (for context): {name if name else 'An experienced professional'}
    """
    
    # 3. Call the Gemini Model
    try:
        response = genai.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=512,
            )
        )
        # Process the model's text response into a list of lines
        return response.text.strip().split('\n')
        
    except APIError as e:
        return [f"An API Error occurred: Please check your API key or network status. Details: {e}"]
    except Exception as e:
        return [f"An unexpected error occurred: {e}"]


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="AI Resume Bullet Generator", layout="centered")
st.title("📄 AI-Powered Resume Bullet Generator")
st.markdown("Craft **impactful, job-specific** resume bullet points using the Gemini AI.")

# Input fields
with st.form("bullet_generator_form"):
    st.header("Your Information")
    user_name = st.text_input("Your Name (Optional, for context):", max_chars=100)
    job_title = st.text_input("Target Job Title (e.g., 'Senior Data Analyst'):", key="job_title")
    
    st.markdown("---")
    
    st.header("Key Skills & Experience")
    skills_input = st.text_area(
        "List your Key Skills (one per line, e.g., Python, SQL, Project Management):",
        height=150,
        key="skills_input"
    )
    
    submitted = st.form_submit_button("🚀 Generate Bullet Points", use_container_width=True)

# Processing and Output
if submitted:
    # Clean and parse the skills list from the text area
    skills_list = [s.strip() for s in skills_input.split('\n') if s.strip()]
    
    if not job_title or not skills_list:
        st.error("❗ Please fill in the **Target Job Title** and **Key Skills** fields.")
    else:
        with st.spinner(f"Gemini is crafting powerful bullets for the '{job_title}' role..."):
            # Call the generation function
            bullets = generate_resume_bullets(user_name, job_title, skills_list)
        
        st.success("Generation Complete! Copy and paste these points into your resume.")
        st.subheader("📋 Your Impactful Resume Bullet Points")
        
        # Display the bullet points as a list
        bullet_html = "<ul>"
        for bullet in bullets:
            # Clean up common LLM output formatting (starting with a hyphen, number, or asterisk)
            cleaned_bullet = bullet.strip()
            # Remove common list prefixes like '1.', '-', or '*'
            cleaned_bullet = cleaned_bullet.lstrip('-*').strip()
            if cleaned_bullet and cleaned_bullet[0].isdigit():
                cleaned_bullet = cleaned_bullet.lstrip('0123456789. ').strip()
                
            if cleaned_bullet:
                 bullet_html += f"<li>{cleaned_bullet}</li>"
        bullet_html += "</ul>"
        
        st.markdown(bullet_html, unsafe_allow_html=True)
        st.caption(f"Powered by {MODEL_NAME}")
