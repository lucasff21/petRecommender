import pandas as pd
import random

NUM_PETS = 500

print("Gerando dataset final de pets...")
pets_data = []
for pet_id in range(1, NUM_PETS + 1):
    pets_data.append({
        'pet_id': pet_id,
        'idade': random.choice(['Filhote', 'Adulto', 'Idoso']),
        'sexo': random.choice(['Macho', 'Fêmea']),
        'porte': random.choice(['Pequeno', 'Médio', 'Grande']),
        'nivel_queda_pelo': random.randint(1, 5),
        'nivel_latido': random.randint(1, 5),
        'sociabilidade_gatos': random.randint(1, 5),
        'sociabilidade_caes': random.randint(1, 5),
        'sociabilidade_criancas': random.randint(1, 5),
        'instinto_guarda': random.randint(1, 5),
        'nivel_energia': random.randint(1, 5),
        'saude': random.choice(['Saudável', 'Tratamento pontual', 'Requer cuidado especial']), 
        'ambiente_adequado': random.choice(['Casa com quintal', 'Apartamento', 'Flexível', 'Casa (qualquer)']), 
        'necessidade_companhia': random.randint(1, 5),
        'treinabilidade': random.randint(1, 5) 
    })

df_pets = pd.DataFrame(pets_data)
df_pets.to_csv('pets_final.csv', index=False, encoding='utf-8')
print("Arquivo 'pets_final.csv' gerado com sucesso!")
print(df_pets.head())