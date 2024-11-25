import streamlit as st
import yaml
import json

intro_page = st.Page("model/Intro.py", title="Introdução", icon="📑")
part1 = st.Page("model/Part1.py", title="Part 1", icon="🗺️")
part2 = st.Page("model/Part2.py", title="Part 2", icon="🗺️")

pg = st.navigation([intro_page, part1, part2])

st.set_page_config(
        page_title="Intro",
        page_icon="Infnet_logo.png",
        layout="wide",
        initial_sidebar_state = "expanded")


#Carregar configurações
#with open('app/config/config.json', 'r') as arquivo:
#        st.session_state['config'] = json.loads(arquivo.read())

#with open('app/config/gemini_config.yaml', 'r') as arquivo:
#        st.session_state['gemini_config'] = yaml.safe_load(arquivo)


pg.run()