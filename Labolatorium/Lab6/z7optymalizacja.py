import numpy as np
from scipy.optimize import linprog

# Koszty transportu
costs = np.array([
    [50, 40, 50, 20],
    [40, 80, 70, 30],
    [60, 40, 70, 80]
])

# Wielkości dostaw z magazynów
supply = np.array([70, 50, 80])

# Zapotrzebowanie odbiorców
demand = np.array([40, 60, 50, 50])

# Liczba magazynów i odbiorców
num_warehouses, num_customers = costs.shape

# Funkcja celu: minimalizacja kosztów transportu
c = costs.flatten()

# Ograniczenia dla podaży (każdy magazyn może dostarczyć maksymalnie Ai ton)
A_eq = np.zeros((num_warehouses + num_customers, num_warehouses * num_customers))
for i in range(num_warehouses):
    A_eq[i, i*num_customers:(i+1)*num_customers] = 1

# Ograniczenia dla popytu (każdy odbiorca musi otrzymać Bj ton)
for j in range(num_customers):
    A_eq[num_warehouses + j, j::num_customers] = 1

# Prawa strona równań (dostępna podaż i zapotrzebowanie)
b_eq = np.concatenate([supply, demand])

# Granice zmiennych (wszystkie wartości >= 0)
bounds = [(0, None) for _ in range(num_warehouses * num_customers)]

# Rozwiązanie problemu
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

# Wyniki
if result.success:
    print("Minimalny koszt transportu:", result.fun)
    solution = result.x.reshape(num_warehouses, num_customers)
    print("Plan transportu (w tonach):")
    print(solution)
else:
    print("Problem nie ma rozwiązania")

