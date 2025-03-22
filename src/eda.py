import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    """Wczytuje dane z pliku JSON do DataFrame"""
    return pd.read_json(file_path)

def descriptive_statistics(df):
    """Oblicza i wyświetla statystyki opisowe dla liczby składników"""
    # Liczba składników
    df['num_ingredients'] = df['ingredients'].apply(len)
    print("Statystyki opisowe dla liczby składników:")
    print(df['num_ingredients'].describe())

    # Liczba składników alkoholowych
    df['num_alcoholic_ingredients'] = df['ingredients'].apply(lambda x: sum(ing['alcohol'] for ing in x))
    print("Statystyki opisowe dla liczby składników alkoholowych:")
    print(df['num_alcoholic_ingredients'].describe())

    # Procent składników alkoholowych
    df['percent_alcoholic'] = df['num_alcoholic_ingredients'] / df['num_ingredients'] * 100
    print("\nStatystyki opisowe dla procentu składników alkoholowych:")
    print(df['percent_alcoholic'].describe())

    # Rozkład kategorii koktajli i szklanek
    print("\nNajczęstsze kategorie koktajli:")
    print(df['category'].value_counts())
    print("\nNajczęstsze typy szklanek:")
    print(df['glass'].value_counts())


def plot_histogram(df):
    """Tworzy histogram liczby składników i zapisuje go do pliku"""
    # Histogram liczby składników
    min_val = int(df['num_ingredients'].min())
    max_val = int(df['num_ingredients'].max())
    
    bins = np.arange(min_val - 0.5, max_val + 1.5, 1.0)
    
    plt.hist(df['num_ingredients'], bins=bins, color='blue', rwidth=0.8)
    plt.title('Rozkład liczby składników')
    plt.xlabel('Liczba składników')
    plt.ylabel('Liczba koktajli')
    
    x_ticks = np.arange(min_val, max_val + 1, 1)
    plt.xticks(x_ticks)
    plt.savefig('../figures/num_ingredients_hist.png')
    plt.close()

    # Histogram liczby składników alkoholowych
    min_val = int(df['num_alcoholic_ingredients'].min())
    max_val = int(df['num_alcoholic_ingredients'].max())
    bins = np.arange(min_val - 0.5, max_val + 1.5, 1.0)
    plt.hist(df['num_alcoholic_ingredients'], bins=bins, color='blue', rwidth=0.8)
    plt.title('Rozkład liczby składników alkoholowych')
    plt.xlabel('Liczba składników alkoholowych')
    plt.ylabel('Liczba koktajli')
    x_ticks = np.arange(min_val, max_val + 1, 1)
    plt.xticks(x_ticks)
    plt.savefig('../figures/num_alcoholic_hist.png')
    plt.close()

    # Histogram procentu składników alkoholowych
    min_val = int(df['percent_alcoholic'].min())
    max_val = int(df['percent_alcoholic'].max())
    bins = np.arange(min_val, 110, 10)
    plt.hist(df['percent_alcoholic'], bins=bins, color='blue', rwidth=0.8)
    plt.title('Rozkład procentu składników alkoholowych')
    plt.xlabel('Procent składników alkoholowych (%)')
    plt.ylabel('Liczba koktajli')
    x_ticks = np.arange(min_val, max_val + 1, 20)
    plt.xticks(x_ticks)
    plt.savefig('../figures/percent_alcoholic_hist.png')
    plt.close()


if __name__ == "__main__":
    df = load_data('../data/cocktail_dataset.json')
    descriptive_statistics(df)
    plot_histogram(df)