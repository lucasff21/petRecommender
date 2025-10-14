import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

def preparar_dados(df_tutores, df_pets):
    """
    Versão final e simplificada da função de preparação de dados.
    Agora que os nomes das colunas já vêm padronizados, não precisamos mais renomeá-los aqui.
    """
    
    # --- Engenharia de Features no DATAFRAME DE TUTORES ---
    # Este passo continua importante para "traduzir" o contexto do tutor em preferências numéricas.
    mapa_criancas = {'Não': 1, 'Crianças maiores': 4, 'Crianças pequenas': 5}
    df_tutores['sociabilidade_criancas'] = df_tutores['tem_criancas'].map(mapa_criancas)
    df_tutores['sociabilidade_caes'] = df_tutores['possui_caes'].apply(lambda x: 5 if x else 3)
    df_tutores['sociabilidade_gatos'] = df_tutores['possui_gatos'].apply(lambda x: 5 if x else 3)
    df_tutores['treinabilidade'] = df_tutores['experiencia_com_pets']
    df_tutores['necessidade_companhia'] = 6 - df_tutores['tempo_disponivel']
    
    # --- Seleção Final das Colunas para o "Match" ---
    # Esta lista define exatamente quais características serão comparadas.
    colunas_de_match = [
        'idade', 'sexo', 'porte', 'nivel_queda_pelo', 'nivel_latido', 
        'sociabilidade_gatos', 'sociabilidade_caes', 'sociabilidade_criancas', 
        'instinto_guarda', 'nivel_energia', 'necessidade_companhia', 'treinabilidade'
    ]
    perfil_pet = df_pets[colunas_de_match]
    perfil_tutor = df_tutores[colunas_de_match] # Usamos df_tutores diretamente, sem renomear!

    # --- Vetorização (O restante do código permanece o mesmo) ---
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
        return pd.Series()
        
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
    
    tutores_vec, pets_vec = preparar_dados(df_tutores_raw.copy(), df_pets_raw.copy())
    
    print("\nIniciando geração de recomendações para TODOS os tutores...")
    todas_as_recomendacoes = []
    total_tutores = len(df_tutores_raw)

    for index, tutor in df_tutores_raw.iterrows():
        tutor_id = tutor['user_id']
        
        if (index + 1) % 100 == 0 or (index + 1) == total_tutores:
            print(f"Processando... {index + 1}/{total_tutores}")
        
        recomendacoes = recomendar_cosseno(tutor_id, tutores_vec, pets_vec, top_n=5)
        
        for rank, (pet_id, score) in enumerate(recomendacoes.items(), 1):
            todas_as_recomendacoes.append({
                'id_tutor': tutor_id,
                'rank_recomendacao': rank,
                'id_pet_recomendado': pet_id,
                'score_similaridade': score
            })

    df_recomendacoes_finais = pd.DataFrame(todas_as_recomendacoes)

    output_filename = 'recomendacoes_cosseno.csv'
    df_recomendacoes_finais.to_csv(output_filename, index=False, encoding='utf-8')
    
    print(f"\nPROCESSO CONCLUÍDO!")
    print(f"Arquivo '{output_filename}' gerado com sucesso.")
    print("\nAmostra das recomendações geradas:")
    print(df_recomendacoes_finais.head(10))

except FileNotFoundError:
    print("ERRO: Certifique-se de que os arquivos 'tutores_final.csv' e 'pets_final.csv' estão na mesma pasta.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")