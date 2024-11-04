import pandas as pd


def unbounded_knapsack(items, max_weight):
    dp = [0] * (max_weight + 1)
    track = [-1] * (max_weight + 1)  # To track the items used

    for i in range(len(items)):
        weight, value = items[i]
        for j in range(weight, max_weight + 1):
            if dp[j] < dp[j - weight] + value:
                dp[j] = dp[j - weight] + value
                track[j] = i  # Track which item was used

    # Reconstruction of the items
    w = max_weight
    selected_items = []

    while w > 0 and track[w] != -1:
        selected_item = track[w]
        selected_items.append(items[selected_item])
        w -= items[selected_item][0]

    return dp[max_weight], selected_items


# Example items and max weight
items = [(8, 12), (7, 10), (3, 4), (2, 3)]
max_weight = 53
max_value, selected_items = unbounded_knapsack(items, max_weight)

# Preparing data for the table
data = {
    "Waga": [item[0] for item in selected_items],
    "Wartość": [item[1] for item in selected_items]
}
df = pd.DataFrame(data)

print(f"Maksymalna wartość: {max_value}")
print("Wybrane przedmioty:")
print(df)
