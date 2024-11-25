import google.generativeai as genai
import streamlit as st

def GetGeminiModel(config : dict):
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
    if 'system_instruction' not in config:
        config['system_instruction'] = None
    if 'safety_settings' not in config:
        config['safety_settings'] = None
    if 'generation_config' not in config:
        config['generation_config'] = None
    model = genai.GenerativeModel(config['model']
                              ,system_instruction = config['system_instruction']
                              ,safety_settings = config['safety_settings']
                              ,generation_config = config['generation_config'])
    return model