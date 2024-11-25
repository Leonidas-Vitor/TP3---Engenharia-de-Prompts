import streamlit as st
import os
from services import GeminiConfig as gc
import tiktoken as tk
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, classification_report

st.header('Parte 1 - TP3 - Engenharia de Prompts para Ciência de Dados')

@st.cache_data
def GetGeminiResponse(config : dict, prompt : str):
    model = gc.GetGeminiModel(config)
    response = model.generate_content(prompt)
    return response

@st.cache_data
def GetSimpsonsData():
    personagens = pd.read_csv('https://api.onedrive.com/v1.0/shares/s!Asuw4D2AHTOZvdYgFKCa8GSNfgBMMQ/root/content', low_memory=False)
    episodios = pd.read_csv('https://api.onedrive.com/v1.0/shares/s!Asuw4D2AHTOZvdYfYJmo6ZipoU1WCA/root/content', low_memory=False)
    locais = pd.read_csv('https://api.onedrive.com/v1.0/shares/s!Asuw4D2AHTOZvdYeqNqB1IZ8F-nLvA/root/content', low_memory=False)
    falas = pd.read_csv('https://api.onedrive.com/v1.0/shares/s!Asuw4D2AHTOZvdZB2KyKP8272EUM_g/root/content', low_memory=False)

    falas['episode_id'] = falas['episode_id'].astype(str)
    falas['character_id'] = falas['character_id'].astype(str)
    falas['location_id'] = falas['location_id'].astype(str)
    personagens['id'] = personagens['id'].astype(str)
    episodios['id'] = episodios['id'].astype(str)
    locais['id'] = locais['id'].astype(str)

    episodios.rename(columns={'id':'episode_id', 'title':'episode_title','original_air_date':'episode_air_date'}, inplace=True)
    personagens.rename(columns={'id':'character_id', 'name':'character_name','normalized_name':'character_normalized_name'}, inplace=True) 
    locais.rename(columns={'id':'location_id','name':'location_name','normalized_name':'location_normalized_name'}, inplace=True)

    #simpsons = pd.merge(falas, episodios, left_on='episode_id', right_on='id', mode='left')
    #simpsons = pd.merge(falas, episodios, on='episode_id')
    #simpsons = pd.merge(simpsons, personagens, on='character_id')
    #simpsons = pd.merge(simpsons, locais, on='location_id')

    simpsons = pd.merge(falas, episodios, left_on='episode_id', right_on='episode_id')
    simpsons = pd.merge(simpsons, personagens, left_on='character_id', right_on='character_id', how='left')
    simpsons = pd.merge(simpsons, locais, left_on='location_id', right_on='location_id', how='left')

    return simpsons

with st.container():        
    st.subheader('5. Base de dados The Simpsons', divider=True)
    st.markdown('''*Baixe a base de dados com os episódios do The Simpsons no Kaggle. 
                Utilize os códigos de referência do curso para combinar todos os arquivos CSVs num único dataset. 
                Utilize a biblioteca tiktoken com a codificação cl100k_base para descrever a quantidade de tokens por episódios e temporada.*
                \n1. Quantos tokens em média tem um episódio? E temporada? Qual foi a temporada e o episódio com mais tokens? Faça uma análise descritiva.
                \n2. Utilize a técnica de Prompt Chaining para fazer uma análise descritiva das avaliações do IMDB e da audiência dos episódios. 
                Justifique os prompts gerados.''')
    
    encoding = tk.get_encoding("cl100k_base")
    df_simpsons = GetSimpsonsData()

    st.write('**Base de dados The Simpsons (Amostra):**')
    st.write(df_simpsons.head())

    st.download_button('Download',df_simpsons.to_csv(index=False),file_name='simpsons.csv',mime='text/csv')

    st.divider()

    cols = st.columns(2)
    with cols[0]:
        st.write('**Por episódio:**')
        df_simpsons_agrupado_epi = df_simpsons.groupby(['episode_id','episode_title']).agg({'raw_text':'sum'}).reset_index()
        df_simpsons_agrupado_epi['tokens'] = df_simpsons_agrupado_epi['raw_text'].apply(lambda x: len(encoding.encode(x)))
        st.metric('Média de tokens por episódio',round(df_simpsons_agrupado_epi['tokens'].mean(),1))
        st.metric('Episódio com mais tokens',df_simpsons_agrupado_epi.loc[df_simpsons_agrupado_epi['tokens'].idxmax()]['episode_title'])

    with cols[1]:
        st.write('**Por temporada:**')
        df_simpsons_agrupado_temp = df_simpsons.groupby(['season']).agg({'raw_text':'sum'}).reset_index()
        df_simpsons_agrupado_temp['tokens'] = df_simpsons_agrupado_temp['raw_text'].apply(lambda x: len(encoding.encode(x)))
        st.metric('Média de tokens por temporada',round(df_simpsons_agrupado_temp['tokens'].mean(),1))
        st.metric('Temporada com mais tokens',df_simpsons_agrupado_temp.loc[df_simpsons_agrupado_temp['tokens'].idxmax()]['season'])

    #-------------------------------------------------------------
    st.divider()
    #-------------------------------------------------------------

    st.dataframe(df_simpsons.sample(5)) 
    df_simpsons_pocket = df_simpsons[['imdb_rating', 'views', 'episode_title', 'raw_text']].copy()
    df_simpsons_pocket = df_simpsons_pocket.groupby(['episode_title']).agg({'imdb_rating':'mean','views':'mean','raw_text':'sum'}).reset_index()

    try:
        epi = df_simpsons_pocket[df_simpsons_pocket['episode_title'] == st.session_state['episode_title']]
    except:
        epi = df_simpsons_pocket.sample(1)
        st.session_state['episode_title'] = epi['episode_title'].values[0]

    st.write('**System Instruction**')

    systemInstruction_5 = '''
        Você é um analista de dados e deve fazer uma análise descritiva das avaliações do IMDB e da audiência dos episódios. 
        Você tem acesso apenas aos dados do seguinte episódio <{epi}>:
    '''
    st.code(systemInstruction_5.format(epi=str(epi['episode_title'].values[0])))

    #systemInstruction_5 = f'''
    #    Você é um analista de dados e deve fazer uma análise descritiva das avaliações do IMDB e da audiência dos episódios de forma muito resumida e breve. 
    #    Você tem acesso apenas aos dados do seguinte episódio {epi.to_json()}:
    #'''

    st.dataframe(epi, use_container_width=True)

    if 'interacoes' not in st.session_state:
        st.session_state['interacoes'] = 0
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    #st.write(st.session_state['history'])

    def WriteHistory(messages):
        try:
            for history in st.session_state['history']:
                role = 'Usuário' if 'Usuário' in history else 'Assistente'
                icon = 'user' if 'Usuário' in history else 'assistant'
                messages.chat_message(icon).write(history[role])
        except:
            st.error('Erro ao escrever histórico')
            pass

    

       # with st.spinner('Aguarde...'):
    cols = st.columns([0.8,0.2])
    with cols[0]:
        messages = st.container(height=300)
        WriteHistory(messages)
        if st.session_state['interacoes'] < 3:
            if prompt := st.chat_input("Chat com LLM"):
                st.session_state['history'].append({'Usuário':prompt})
                messages.chat_message("user").write(prompt)
                response = GetGeminiResponse({'model':'gemini-1.5-flash', 
                                            'generation_config':{'candidate_count':1,'max_output_tokens':500,'temperature':0.5},
                                            'system_instruction':systemInstruction_5.format(epi=epi.to_json())},json.dumps(st.session_state['history'], ensure_ascii=False, indent=4))
                messages.chat_message("assistant").write(f"Echo: {response.text}")

                st.session_state['history'].append({'Assistente':response.text})
                st.session_state['interacoes'] += 1
        else:
            st.write('Limite atingido, reinicia o chat para continuar')
    with cols[1]:
        st.write('Esse chat tem o limite de 3 interações, ao atingir o limite, reinicie o chat para continuar')
        st.write(f'Interações: {st.session_state["interacoes"]}/3')
        if st.button('Reiniciar'):
            st.session_state['interacoes'] = 0
            st.session_state['history'].clear()
            st.rerun()
        if st.session_state['interacoes'] == 3:
            st.warning('Limite atingido, reinicia o chat para continuar')

    st.divider()
    st.markdown('**Prompt Chaining**')
    st.write('''R.: O primeiro prompt foi utilizado para que o LLM entende-se o conteúdo do episódio e o objetivo da análise. 
            O segundo prompt foi utilizado para que o LLM fizesse uma análise se o episódio foi bem recebido, se baseando na audiência e nota IMDB. 
            O terceiro prompt foi utilizado para concluir a análise, com base nas informações obtidas nos prompts anteriores e 
            apresentar uma teoria sobre a recepção do episódio.''')	
    cols = st.columns(3)
    with cols[0]:
        st.image('data/images/tp5_A.png',use_container_width=True)
    with cols[1]:
        st.image('data/images/tp5_B.png',use_container_width=True)
    with cols[2]:
        st.image('data/images/tp5_C.png',use_container_width=True)

#-------------------------------------------------------------

with st.container():        
    st.subheader('6. Classificação de Sentimento com Few-Shot Learning', divider=True)
    st.markdown('''*Implemente um modelo de classificação de sentimentos em Python para categorizar trechos de diálogo dos Simpsons como 
                “Positivo”, “Neutro” ou “Negativo”. Use a técnica de few-shot learning, incluindo 5 exemplos por categoria no prompt. 
                Selecione o episódio número 92 (episode_id) da temporada 5 (episode_season). Utilize a técnica de batch-prompting para 
                classificar múltiplas falas num único prompt. Responda às perguntas:*
                \n1.Quantas chamadas ao LLM foram necessárias?
                \n2.Qual é a distribuição de fala por categoria?
                \n3.Avaliando 5 falas de cada classe, qual é a acurácia do modelo?
                \n4.Qual foi a precisão do modelo para cada classe?''')
    
    st.divider()


    falas = df_simpsons[df_simpsons['episode_id'] == '92'][['raw_text']].sample(5, random_state=42)

    st.write('**Falas selecionadas do episódio 92:**')
    st.dataframe(falas,use_container_width=True)

    st.write('**System Instruction**')
    systemInstruction_6 = '''
    Sua tarefa é classificar o sentimento de cada falas de personagens como Positivo, Neutro ou Negativo.
    Para isso, considere os sentimentos expressos nas falas e classifique-as de acordo com a categoria correta. 
    Aqui estão alguns exemplos para ajudar você a entender:

    Fala: "Eu odeio quando isso acontece."
    Classificação: Negativo
    Fala: "Você é horrível!"
    Classificação: Negativo
    Fala: "Vou pro Moe's"
    Classificação: Negativo
    Fala: "Implicaram comigo na escola"
    Classificação: Negativo
    Fala: "Que dever de casa?"
    Classificação: Negativo

    Fala: "Olhe os passarinhos!"
    Classificação: Neutro
    Fala: "Veja o que eu fiz!"
    Classificação: Neutro
    Fala: "Oi"
    Classificação: Neutro
    Fala: "Vem sempre aqui?"
    Classificação: Neutro
    Fala: "Porco aranha, porco aranha, faz tudo o que um porco faz."
    Classificação: Neutro

    Fala: "Você é incrível!"
    Classificação: Positivo
    Fala: "Eu te amo!"
    Classificação: Positivo
    Fala: "Vai ficar tudo bem."
    Classificação: Positivo
    Fala: "Parabéns!"
    Classificação: Positivo
    Fala: "Liza vai tocar saxofone."
    Classificação: Positivo

    A resposta deve ser no formato Json conforme o exemplo: 
    {'Fala': <Fala> , 'Classificação': <Positivo/Neutro/Negativo>}
    A resposta deve estar pronta para ser convertida em formato json sem erros ou processos adicionais.
    Agora, classifique as falas de acordo com a categoria correta.
    '''

    st.code(systemInstruction_6)

    classification = {}
    with st.spinner('Aguarde...'):
        response = GetGeminiResponse(
            {'model':'gemini-1.5-flash', 
            'system_instruction':systemInstruction_6,
            'generation_config':{'candidate_count':1,'max_output_tokens':500,'temperature':0.5}},
            falas.to_json(orient='records', lines=True))
        st.write('**Resposta do modelo:**')
        classification = response.text.replace("```","").replace("json","")
        st.code(classification)

    #histograma das classificações
    st.write('**Distribuição de fala por categoria:**')
    fig = go.Figure(data=[go.Bar(x=['Positivo','Neutro','Negativo'], y=[classification.count('Positivo'),classification.count('Neutro'),classification.count('Negativo')])])
    fig.update_layout(title_text='Classificação das Falas', xaxis_title='Classificação', yaxis_title='Quantidade')
    st.plotly_chart(fig)

    st.write('''**R.:**''')
    st.write('1. Foi necessário apenas 1 chamada ao LLM foi necessária.')
    st.write('2. A distribuição de fala por categoria foi:')
    st.image('data/images/tp6.png',use_container_width=True)
    st.write('3. Considerando: ')
    st.markdown( '''
                | Fala | Classificação |
                | --- | --- |
                | 1ª | Errou |
                | 2ª | Acertou |
                | 3ª | "Acertou" |
                | 4ª | Acertou |
                | 5ª | Acertou |
                ''')
    y_true = ["negativo", "negativo", "negativo", "positivo", "negativo", "neutro"]
    y_pred = ["neutro", "negativo", "negativo", "positivo", "negativo", "neutro"]
    acuracia = accuracy_score(y_true, y_pred)
    st.write(f'A acuária do modelo foi de {round(acuracia,1)}')
    precisao = precision_score(y_true, y_pred, average=None, labels=["positivo", "neutro", "negativo"])
    st.write(f'4. A precisão do modelo foi de:')
    st.write(f'Positivo: {round(precisao[0],1)}')
    st.write(f'Neutro: {round(precisao[1],1)}')
    st.write(f'Negativo: {round(precisao[2],1)}')

#-------------------------------------------------------------

with st.container():        
    st.subheader('7. Resumo Episódio', divider=True)
    st.markdown('''*Assista ao episódio “Homer, o vigilante” (ou leia as falas dos personagens), número 92 (episode_id) da temporada 5 (episode_season) 
                e faça um resumo de aproximadamente 500 tokens (meça a quantidade usando o modelo do exercício 5), explicando o que acontece e como 
                termina o episódio.*''')

#-------------------------------------------------------------

with st.container():        
    st.subheader('8. Resumos Complexos com Chunks de Texto', divider=True)
    st.markdown('''*Crie um prompt para resumir o episódio número 92 (episode_id) da temporada 5 (episode_season) usando o princípio de divisão para 
                contornar limitações de tokens. Utilize o processo de chunks para separar o episódio em janelas de 100 falas, com sobreposição de 25 
                falas por janela. Utilize o LLM para resumir cada um dos chunks. Posteriormente, crie um segundo prompt com os resumos dos chunks 
                instruindo o LLM a gerar o resumo final. Quantos chunks foram necessários? Avalie o resultado do resumo final e de cada chunk quanto à 
                veracidade e coerência.*''')

#-------------------------------------------------------------

with st.container():        
    st.subheader('9. Avaliação de Resumos de LLMs', divider=True)
    st.markdown('''*Utilize as métricas BLEU e ROUGE para comparar os resultados dos prompts do exercício 8 com o seu resumo, feito no exercício 7 
                (utilize qualquer LLM para traduzir entre inglês e portugês se necessário). Aplique as métricas, tanto ao resumo final, quanto ao 
                resumo de cada chunk. Interprete as métricas considerando que o seu resumo é o gabarito. Os resumos (final e de cada chunk) convergem? 
                Quais informações foram omitidas entre os dois resumos?*''')

#-------------------------------------------------------------

with st.container():        
    st.subheader('10. Chain of Thoughts para Codificação', divider=True)
    st.markdown('''*Exporte o resultado da análise de sentimento do exercício 6 para um arquivo CSV. Agora, construa uma série de prompts com a técnica 
                chain of thoughts para construir uma aplicação streamlit que faça a leitura do resultado da análise de sentimento e faça um gráfico de 
                pizza mostrando a proporção de falas de categoria do episódio. Divida o problema em três prompts e execute o código final. O LLM foi capaz 
                de implementar a aplicação? Qual foi o objetivo de cada prompt?*''')