import pulp

# Dane wejściowe
tkaniny = ['T1', 'T2', 'T3', 'T4', 'T5']
krosna = ['K1', 'K2', 'K3']
wydajnosci = {
    'K1': {'T1': 5, 'T2': 10, 'T3': 8, 'T4': 12, 'T5': 6},
    'K2': {'T1': 7, 'T2': 7, 'T3': 12, 'T4': 10, 'T5': 8},
    'K3': {'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 9}
}
max_czas_pracy = {'K1': 600, 'K2': 840, 'K3': 720}
zapotrzebowanie = {'T1': 1120, 'T2': 1260, 'T3': 1800, 'T4': 1200, 'T5': 720}

# Problem minimalizacji czasu pracy
prob = pulp.LpProblem("MinimizacjaCzasuPracy", pulp.LpMinimize)

# Zmienne decyzyjne
x = pulp.LpVariable.dicts("produkcja", ((k, t) for k in krosna for t in tkaniny), lowBound=0, cat='Continuous')

# Funkcja celu: minimalizacja całkowitego czasu pracy
prob += pulp.lpSum((1 / wydajnosci[k][t]) * x[k, t] for k in krosna for t in tkaniny)

# Ograniczenia: maksymalny czas pracy krosna
for k in krosna:
    prob += pulp.lpSum((1 / wydajnosci[k][t]) * x[k, t] for t in tkaniny) <= max_czas_pracy[k]

# Ograniczenia: zapotrzebowanie na tkaniny
for t in tkaniny:
    prob += pulp.lpSum(x[k, t] for k in krosna) >= zapotrzebowanie[t]

# Rozwiązanie problemu
prob.solve()

# Wyniki
print(f"Status: {pulp.LpStatus[prob.status]}")
for k in krosna:
    for t in tkaniny:
        print(f"Krosno {k}, Tkanina {t}: {x[k, t].varValue} m")

print(f"Minimalny całkowity czas pracy: {pulp.value(prob.objective)} godzin")
