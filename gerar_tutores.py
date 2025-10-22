import pandas as pd
import random

NUM_TUTORES = 10

print("Gerando dataset final de tutores...")
tutores_data = []
for user_id in range(1, NUM_TUTORES + 1):
    tutores_data.append(
        {
            "user_id": user_id,
            "moradia": random.choice(["Casa com quintal", "Casa sem quintal", "Apartamento"]),
            "ambiente_casa": random.choice(["Calmo e silencioso", "Pouco movimento", "Movimentado com visitas"]),
            "possui_caes": random.choice([True, False]), 
            "possui_gatos": random.choice([True, False]),  
            "tem_criancas": random.choice(["Não", "Crianças pequenas", "Crianças maiores"]),  
            "experiencia_com_pets": random.randint(1, 5),
            "tempo_disponivel": random.randint(1, 5),
            "disposicao_necessidades_especiais": random.choice([True, False]),
            "idade": random.choice(["Filhote", "Adulto", "Idoso", "Indiferente"]),
            "porte": random.choice(["Pequeno", "Médio", "Grande", "Indiferente"]),
            "sexo": random.choice(["Macho", "Fêmea", "Indiferente"]),
            "nivel_energia": random.randint(1, 5),
            "instinto_guarda": random.randint(1, 5),
            "nivel_queda_pelo": random.randint(1, 5),
            "nivel_latido": random.randint(1, 5),
        }
    )

df_tutores = pd.DataFrame(tutores_data)
df_tutores.to_csv("tutores_final.csv", index=False, encoding="utf-8")
print("Arquivo 'tutores_final.csv' gerado com sucesso!")
print(df_tutores.head())
