import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Configuração inicial
warnings.filterwarnings("ignore")
plt.style.use('ggplot') 

# Carregar dataset (uma vez só)
df = pd.read_csv('Warehouse_and_Retail_Sales.csv')
df['Data'] = pd.to_datetime({'year': df['YEAR'], 'month': df['MONTH'], 'day': 1})

# ==============================================================================
# --- ANÁLISE 1: VINHO (WINE) ---
# ==============================================================================
print("\n" + "="*50)
print("INICIANDO ANÁLISE 1: VINHO (WINE)")
print("="*50)

df_vinho = df[df['ITEM TYPE'] == 'WINE']
serie_vinho = df_vinho.groupby('Data')['RETAIL SALES'].sum().asfreq('MS').fillna(0)

# 1. Traçar Série (Vinho)
plt.figure(figsize=(12, 5))

# Adicionei marker='.' para mostrar cada ponto traçado (fica mais técnico)
plt.plot(serie_vinho.index, serie_vinho, label='Dados Observados', color='#800020', marker='.', linestyle='-')

plt.title('1. Traçado da Série Temporal: Vendas de VINHO (WINE)', fontsize=14, fontweight='bold')
# -------------------------------------------

plt.xlabel('Eixo Temporal (Mensal)')
plt.ylabel('Volume de Vendas')
plt.legend()
plt.grid(True, alpha=0.3) # O grid ajuda a ver o traçado melhor
plt.show()

# 2. Decomposição (Vinho)
decomp_vinho = seasonal_decompose(serie_vinho, model='additive')
fig = decomp_vinho.plot()
fig.set_size_inches(10, 8)
fig.suptitle('2. Decomposição (Vinho)', fontsize=14)
plt.tight_layout()
plt.show()

# 3. Previsão Longa (Vinho)
modelo_vinho = ExponentialSmoothing(serie_vinho, trend='add', seasonal='add', seasonal_periods=12).fit()
meses_futuros = 72 # 6 anos (até 2026)
prev_vinho = modelo_vinho.forecast(steps=meses_futuros)

plt.figure(figsize=(14, 7))
plt.plot(serie_vinho.index, serie_vinho, label='Histórico', color='#800020')
plt.plot(prev_vinho.index, prev_vinho, label='Projeção Estendida (até 2026)', color='red', linestyle='--')
plt.title('3. Projeção de Cenário: VINHO (até 2026)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Média de Vendas Mensal (Vinho): {serie_vinho.mean():.2f}")

# ==============================================================================
# --- ANÁLISE 2: CERVEJA (BEER) ---
# ==============================================================================
print("\n" + "="*50)
print("INICIANDO ANÁLISE 2: CERVEJA (BEER)")
print("="*50)

df_beer = df[df['ITEM TYPE'] == 'BEER']
serie_beer = df_beer.groupby('Data')['RETAIL SALES'].sum().asfreq('MS').fillna(0)

# 1. Traçar Série (Cerveja)
plt.figure(figsize=(12, 5))

plt.plot(serie_beer.index, serie_beer, label='Dados Observados', color='#FF8C00', marker='.', linestyle='-')

plt.title('1. Traçado da Série Temporal: Vendas de CERVEJA (BEER)', fontsize=14, fontweight='bold')
# -------------------------------------------

plt.xlabel('Eixo Temporal (Mensal)')
plt.ylabel('Volume de Vendas')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()