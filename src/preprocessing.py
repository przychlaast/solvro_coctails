import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder
import random
import copy

def encode_categorical(df):
    """Koduje cechy kategoryczne (category, glass, tags) na wartości numeryczne"""
    df_encoded = pd.get_dummies(df, columns=['category', 'glass'], prefix=['cat', 'glass'])
    
    df['tags'] = df['tags'].apply(lambda x: [] if x is None else x)
    
    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(df['tags'])
    df_tags = pd.DataFrame(tags_encoded, columns=mlb.classes_)
    return pd.concat([df_encoded, df_tags], axis=1), mlb.classes_


def augment_data(df, len_new_samples=1000):
    """Augmentacja danych poprzez tworzenie nowych koktajli"""
    original_size = len(df)
    new_samples = []
    
    category_ingredients = {}
    for _, row in df.iterrows():
        category = row['category']
        if category not in category_ingredients:
            category_ingredients[category] = []
        
        for ing in row['ingredients']:
            if ing not in category_ingredients[category]:
                category_ingredients[category].append(ing)
    
    for i in range(len_new_samples):
        sample1, sample2 = df.sample(2).iloc
        
        new_cocktail = {}
        new_cocktail['name'] = f"Mix {sample1['name']} i {sample2['name']}"
        new_cocktail['category'] = sample1['category']
        new_cocktail['glass'] = random.choice([sample1['glass'], sample2['glass']])
        
        new_ingredients = []
        all_ingredients = sample1['ingredients'] + sample2['ingredients']

        ingredient_names = set()
        for ing in all_ingredients:
            if ing.get('name', '') not in ingredient_names:
                ingredient_names.add(ing.get('name', ''))
                new_ingredients.append(ing)
        
        if len(new_ingredients) < 3 and new_cocktail['category'] in category_ingredients:
            available_ingredients = category_ingredients[new_cocktail['category']]
            while len(new_ingredients) < 3 and available_ingredients:
                random_ing = random.choice(available_ingredients)
                ing_name = random_ing.get('name', '')
                if ing_name not in ingredient_names:
                    ingredient_names.add(ing_name)
                    new_ingredients.append(copy.deepcopy(random_ing))
        
        new_cocktail['ingredients'] = new_ingredients
        
        all_tags = list(set(sample1['tags'] + sample2['tags']))
        new_cocktail['tags'] = all_tags
        
        new_samples.append(new_cocktail)
    
    augmented_df = pd.concat([df, pd.DataFrame(new_samples)], ignore_index=True)
    augmented_df = create_features(augmented_df)
    
    print(f"Zbiór danych został rozszerzony z {original_size} do {len(augmented_df)} próbek.")
    return augmented_df


def scale_features(df, feature_cols):
    """Standaryzuje wybrane cechy numeryczne"""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    return scaled_features, scaler


def load_and_preprocess_data(file_path):
    """Wczytuje dane z pliku JSON i wykonuje podstawowy preprocessing"""
    df = pd.read_json(file_path)
    df = handle_missing_values(df)
    df = create_features(df)
    df = normalize_names(df)
    
    return df


def handle_missing_values(df):
    """Obsługuje brakujące wartości w danych"""

    df['tags'] = df['tags'].apply(lambda x: ['nieokreslone'] if x is None else x)
    df['category'] = df['category'].fillna('inna')
    df['glass'] = df['glass'].fillna('inna')
    
    return df


def normalize_names(df):
    """Normalizuje nazwy składników, kategorii i szklanek"""
    df['ingredients'] = df['ingredients'].apply(normalize_ingredients)
    df['category'] = df['category'].str.lower().str.strip()
    df['glass'] = df['glass'].str.lower().str.strip()
    
    return df


def normalize_ingredients(ingredients_list):
    """Normalizuje nazwy składników w liście"""
    for ing in ingredients_list:
        if 'name' in ing:
            ing['name'] = ing['name'].lower().strip()
    return ingredients_list


def create_features(df):
    """Tworzy nowe cechy na podstawie istniejących danych"""
    df['num_ingredients'] = df['ingredients'].apply(len)
    df['num_alcoholic_ingredients'] = df['ingredients'].apply(lambda x: sum(ing.get('alcohol', 0) for ing in x))
    df['percent_alcoholic'] = (df['num_alcoholic_ingredients'] / df['num_ingredients'] * 100).fillna(0)
    df['num_tags'] = df['tags'].apply(len)
    
    return df


def encode_categorical_features(df):
    """Koduje cechy kategoryczne za pomocą one-hot encoding"""
    encoder_category = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    category_encoded = encoder_category.fit_transform(df[['category']])
    category_df = pd.DataFrame(category_encoded, columns=encoder_category.get_feature_names_out(['category']))
    
    encoder_glass = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    glass_encoded = encoder_glass.fit_transform(df[['glass']])
    glass_df = pd.DataFrame(glass_encoded, columns=encoder_glass.get_feature_names_out(['glass']))
    
    result_df = pd.concat([df.reset_index(drop=True), category_df, glass_df], axis=1)
    
    return result_df, encoder_category, encoder_glass


if __name__ == "__main__":
    df = load_and_preprocess_data('../data/cocktail_dataset.json')
    print(f"Zbiór danych po preprocessingu: {df.shape}")

    augmented_df = augment_data(df, len_new_samples=500)
    encoded_df, _, _ = encode_categorical_features(augmented_df)

    print(f"Zbiór danych po kodowaniu cech: {encoded_df.shape}")
    encoded_df.to_json('../data/processed_cocktails.json', orient='records')