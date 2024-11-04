import numpy as np
import pandas as pd

# Dane wejściowe
cost = np.array([[50, 40, 50, 20],
                 [40, 80, 70, 30],
                 [60, 40, 70, 80]], dtype=float)  # Konwersja na typ float

supply = np.array([70, 50, 80])
demand = np.array([40, 40, 60, 50])

# Inicjalizacja macierzy wyników
result = np.zeros(cost.shape)


# Funkcja zachłanna do minimalizacji kosztów transportu
def greedy_transport(cost, supply, demand):
    total_cost = 0
    while np.any(demand > 0):
        min_cost_idx = np.unravel_index(np.argmin(cost), cost.shape)
        i, j = min_cost_idx

        # Ilość towaru do przewiezienia
        amount = min(supply[i], demand[j])

        # Aktualizacja wyników
        result[i, j] = amount
        supply[i] -= amount
        demand[j] -= amount
        total_cost += amount * cost[i, j]

        # Ustawienie wysokiego kosztu, aby uniknąć ponownego wyboru tej samej ścieżki
        cost[i, j] = np.inf

        # Jeśli zapotrzebowanie lub dostępność są zerowe, ustaw koszty transportu na nieskończoność
        if supply[i] == 0:
            cost[i, :] = np.inf
        if demand[j] == 0:
            cost[:, j] = np.inf

    return total_cost


# Obliczenie minimalnego kosztu transportu
total_cost = greedy_transport(cost.copy(), supply.copy(), demand.copy())

# Wynik jako DataFrame dla czytelności
result_df = pd.DataFrame(result, columns=['O1', 'O2', 'O3', 'O4'], index=['M1', 'M2', 'M3'])

print(f"Minimalny całkowity koszt transportu: {total_cost}")
print("Plan przewozu paliwa:")
print(result_df)
