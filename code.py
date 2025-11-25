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

