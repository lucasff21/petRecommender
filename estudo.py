import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

numeros = [10, 20, 30, 40, 50]

soma_total = 0


for numero in numeros:
    soma_total = numero + soma_total

print(f"a lista é: {numeros}")
print(f"a soma total é {soma_total}")


##################################################
numeros = [15, 8, 29, 42, 11]
maior_verdadeiro = 0

for numero in numeros:
    if numero > maior_verdadeiro:
        maior_verdadeiro = numero

print(f"o maior valor é: {maior_verdadeiro}")        

##################################################
lista_com_duplicados = [1, 2, 3, 2, 4, 5, 4, 1, 6]
lista_unica = []

for numero in lista_com_duplicados:
    if numero not in lista_unica:
        lista_unica.append(numero)

print(f"Lista original: {lista_com_duplicados}")
print(f"Lista com itens únicos: {lista_unica}")

##################################################

numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
numeros_pares = []

for numero in numeros:
    if numero % 2 == 0:
        numeros_pares.append(numero)

print(f"Lista com numeros pares: {numeros_pares}")
