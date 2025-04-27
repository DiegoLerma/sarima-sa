# %% [markdown]
# Actividad IMF

# %% [markdown]
# # 1. Exploracion y deteccion de d y D

# %%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# %% [markdown]
# ## 1.1 Cargar y preparar

# %%

df = pd.read_csv("../data/electric_production.csv", parse_dates=["DATE"], index_col="DATE")
ts = df["IPG2211A2N"].asfreq("MS")

# %% [markdown]
# ## 1.2 Exploracion inicial

# %%
df.head()

# %%
df.describe()

# %%
df.info()

# %%
df.isnull().sum()

# %% [markdown]
# ## 1.3 Visualización rápida

# %%
ts.plot(title="Producción eléctrica mensual (1985-2018)", figsize=(10,4))
plt.show()

# %% [markdown]
# ## 1.4 Descomposición estacional
# 

# %%
decomp = seasonal_decompose(ts, period=12, model="additive")
decomp.plot(); plt.show()

# %% [markdown]
# ## 1.5 Tests de estacionariedad (nivel)

# %%
def adf_test(series, title=''):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"{title}  ADF p-value: {result[1]:.4f}")

adf_test(ts, "Original")
adf_test(ts.diff().dropna(), "d=1")
adf_test(ts.diff(12).dropna(), "D=1 (solo)")
adf_test(ts.diff().diff(12).dropna(), "d=1, D=1")

# %% [markdown]
# # 2. Espacio de búsqueda y grafo de vecindad

# %%
import itertools
import random

# Dominios de búsqueda
d_candidates = [0, 1, 2]                 # ±1 alrededor de 1
D_candidates = [0, 1, 2]
rng_0_7      = range(0, 8)               # p, q, P, Q ≤ 7
m = 12                                    

# Un nodo es una tupla: (p, d, q, P, D, Q)
def is_valid(node):
    p,d,q,P,D,Q = node
    return (d in d_candidates) and (D in D_candidates)

def get_neighbors(node):
    neigh = []
    bounds = [(0,7),(0,2),(0,7),(0,7),(0,2),(0,7)]
    for idx,(lo,hi) in enumerate(bounds):
        for delta in (-1,1):
            new_val = node[idx] + delta
            if lo <= new_val <= hi:
                new = list(node)
                new[idx] = new_val
                if is_valid(tuple(new)):
                    neigh.append(tuple(new))
    return neigh

# %%
start_node = (1, 1, 1,   # p,d,q
              0, 1, 0)   # P,D,Q

# %% [markdown]
# # 3. Función de coste

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error
import numpy as np

# Separación train / validación (últimos 50 meses)
train, val_set = ts[:-50], ts[-50:]

def score(node):
    p,d,q,P,D,Q = node
    try:
        model = SARIMAX(train,
                        order=(p,d,q),
                        seasonal_order=(P,D,Q,m),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                       ).fit(disp=False)
        pred  = model.forecast(steps=50)
        rmse  = mean_squared_error(val_set, pred, squared=False)

        # Test Ljung-Box a 6 rezagos
        p_lb = acorr_ljungbox(model.resid, lags=[6],
                              return_df=True).iloc[-1,0]
        if p_lb < 0.05:      # residuos autocorrelados ⇒ penalizar
            return np.inf
        return rmse
    except Exception:        # problemas de convergencia
        return np.inf


# %% [markdown]
# # 4. Algoritmo Simulated Annealing

# %%
def simulated_annealing(start, iterations=200, T0=10, alpha=0.95):
    current = best = start
    curr_cost = best_cost = score(current)
    T = T0
    history = [(0, current, curr_cost)]

    for k in range(1, iterations+1):
        neighbors = get_neighbors(current)
        if not neighbors:        # sin vecinos válidos → reinicia
            current, curr_cost = start, score(start)
            continue

        neighbor = random.choice(get_neighbors(current))
        neigh_cost = score(neighbor)
        delta = neigh_cost - curr_cost

        # Criterio de aceptación
        if delta < 0 or np.exp(-delta/T) > random.random():
            current, curr_cost = neighbor, neigh_cost

        if curr_cost < best_cost:
            best, best_cost = current, curr_cost

        history.append((k, current, curr_cost))
        T *= alpha                     # enfriamiento exponencial

    return best, best_cost, history


# %%
best_node, best_rmse, path = simulated_annealing(start_node)
print(f"Mejor configuración: {best_node} – RMSE = {best_rmse:.3f}")

# %% [markdown]
# # 5. Entrenamiento final y validación extendida

# %%
p,d,q,P,D,Q = best_node
final_model = SARIMAX(ts, order=(p,d,q),
                      seasonal_order=(P,D,Q,m),
                      enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False)

# 12-meses de pronóstico hacia el futuro
fcast = final_model.get_forecast(steps=12)
ci = fcast.conf_int()

ax = ts.plot(label="observado", figsize=(10,4))
fcast.predicted_mean.plot(ax=ax, label="pronóstico")
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
ax.legend(); plt.show()

print(final_model.summary())


# %% [markdown]
# # Diagnostico de resultados

# %%
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 1. ACF y PACF de residuos hasta lag 30
sm.graphics.tsa.plot_acf(final_model.resid, lags=30, zero=False)
plt.title("ACF residuos"); plt.show()
sm.graphics.tsa.plot_pacf(final_model.resid, lags=30, zero=False)
plt.title("PACF residuos"); plt.show()

# 2. Ljung-Box acumulado (lags 1-6)
lb = sm.stats.diagnostic.acorr_ljungbox(final_model.resid, lags=[6],
                                        return_df=True)
print(lb)

# 3. Q-Q y histograma
sm.qqplot(final_model.resid, line="s")
plt.title("Q-Q residuos"); plt.show()
plt.hist(final_model.resid, bins=20); plt.title("Histograma residuos"); plt.show()


# %% [markdown]
# ### Resultados finales
# 
# * **Modelo óptimo (SA):** SARIMA(1,1,1)×(0,1,0,12)  
# * **RMSE (validación 50 meses):** 4.27  
# * **Ljung-Box(6) p-value:** 0.136  ✓  
# * **Conclusión:** El modelo captura la estructura anual y la tendencia
#   de la producción eléctrica con error medio absoluto ≈ **4.3 GWh**.
#   Las colas residuales son algo pesadas; se proponen dos mejoras
#   futuras: aplicar transformación logarítmica y estimar intervalos
#   mediante bootstrap.
# 


