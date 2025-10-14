import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

def preparar_dados(df_tutores, df_pets):
    """
    Versão final da função de preparação de dados. Inclui:
    1. Engenharia de Features: "Traduz" as respostas do tutor para serem compatíveis com as do pet.
    2. Vetorização: Transforma tudo em vetores numéricos para o cálculo do cosseno.
    """
    
    # --- 1. ENGENHARIA DE FEATURES NO DATAFRAME DE TUTORES ---
    # O objetivo é criar colunas no df_tutores que correspondam diretamente às dos pets.
    
    # Mapeamento para sociabilidade com crianças
    mapa_criancas = {'Não': 1, 'Crianças maiores': 4, 'Crianças pequenas': 5}
    df_tutores['sociabilidade_criancas'] = df_tutores['tem_criancas'].map(mapa_criancas)

    # Mapeamento para sociabilidade com cães e gatos (se já tem, precisa de um pet sociável)
    df_tutores['sociabilidade_caes'] = df_tutores['possui_caes'].apply(lambda x: 5 if x else 3) # 5 se sim, 3 (neutro) se não
    df_tutores['sociabilidade_gatos'] = df_tutores['possui_gatos'].apply(lambda x: 5 if x else 3) # 5 se sim, 3 (neutro) se não

    # Mapear outras colunas que não têm correspondência direta mas podemos inferir
    df_tutores['treinabilidade'] = df_tutores['experiencia_com_pets'] # Assumimos que tutores experientes preferem cães treináveis
    df_tutores['necessidade_companhia'] = 6 - df_tutores['tempo_disponivel'] # Lógica inversa: quanto menos tempo disponível, menor a necessidade de companhia desejada

    # --- 2. PADRONIZAÇÃO DOS NOMES DAS COLUNAS RESTANTES ---
    mapa_renomear_tutor = {
        'pref_idade': 'idade',
        'pref_porte': 'porte',
        'pref_sexo': 'sexo',
        'tolerancia_pelo': 'nivel_queda_pelo',
        'tolerancia_latido': 'nivel_latido',
        'nivel_energia_desejado': 'nivel_energia',
        'necessidade_guarda_desejada': 'instinto_guarda'
    }
    df_tutores_final = df_tutores.rename(columns=mapa_renomear_tutor)

    # --- 3. SELEÇÃO FINAL DAS COLUNAS PARA O MATCH ---
    # Agora selecionamos o mesmo conjunto de colunas para ambos
    colunas_de_match = [
        'idade', 'sexo', 'porte', 'nivel_queda_pelo', 'nivel_latido', 
        'sociabilidade_gatos', 'sociabilidade_caes', 'sociabilidade_criancas', 
        'instinto_guarda', 'nivel_energia', 'necessidade_companhia', 'treinabilidade'
    ]
    perfil_pet = df_pets[colunas_de_match]
    perfil_tutor = df_tutores_final[colunas_de_match]

    # --- 4. VETORIZAÇÃO ---
    colunas_categoricas = perfil_pet.select_dtypes(include=['object', 'category']).columns.tolist()
    colunas_escala = perfil_pet.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('escala', MinMaxScaler(), colunas_escala),
            ('categoria', OneHotEncoder(handle_unknown='ignore', sparse_output=False), colunas_categoricas)
        ],
        remainder='drop'
    )
    
    preprocessor.fit(perfil_pet)
    
    pets_vetorizados_np = preprocessor.transform(perfil_pet)
    tutores_vetorizados_np = preprocessor.transform(perfil_tutor)
    
    feature_names = preprocessor.get_feature_names_out()
    pets_vec = pd.DataFrame(pets_vetorizados_np, index=df_pets['pet_id'], columns=feature_names)
    tutores_vec = pd.DataFrame(tutores_vetorizados_np, index=df_tutores['user_id'], columns=feature_names)
    
    print("Vetorização concluída com sucesso!")
    return tutores_vec, pets_vec

def recomendar_cosseno(id_tutor, df_tutores_vec, df_pets_vec, top_n=5):
    if id_tutor not in df_tutores_vec.index:
        return "ID do tutor não encontrado."
        
    vetor_tutor = df_tutores_vec.loc[id_tutor].values.reshape(1, -1)
    scores_similaridade = cosine_similarity(vetor_tutor, df_pets_vec)
    scores_pets = pd.Series(scores_similaridade[0], index=df_pets_vec.index)
    pets_recomendados = scores_pets.sort_values(ascending=False)
    return pets_recomendados.head(top_n)

# --- BLOCO PRINCIPAL DE EXECUÇÃO ---
try:
    df_tutores_raw = pd.read_csv('tutores_final.csv')
    df_pets_raw = pd.read_csv('pets_final.csv')
    print("Datasets carregados.")
    
    tutores_vec, pets_vec = preparar_dados(df_tutores_raw, df_pets_raw)
    
    id_do_tutor_para_teste = random.choice(df_tutores_raw['user_id'])
    
    recomendacoes = recomendar_cosseno(id_do_tutor_para_teste, tutores_vec, pets_vec)
    
    print(f"\n==============================================================================")
    print(f"--- Recomendações para o Tutor #{id_do_tutor_para_teste} ---")
    print(f"==============================================================================")
    print("\n>>> Perfil do Tutor:")
    print(df_tutores_raw[df_tutores_raw['user_id'] == id_do_tutor_para_teste].iloc[0].to_string())
    
    print("\n\n>>> Top 5 Pets Recomendados:")
    for pet_id, score in recomendacoes.items():
        print(f"\n--- Pet ID: {pet_id} (Score de Similaridade: {score:.4f}) ---")
        print(df_pets_raw[df_pets_raw['pet_id'] == pet_id].iloc[0].to_string())

except FileNotFoundError:
    print("ERRO: Certifique-se de que os arquivos 'tutores_final.csv' e 'pets_final.csv' estão na mesma pasta.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")