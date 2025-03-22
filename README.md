# Klasteryzacji Koktajli

## Opis
Projekt ma na celu analizę klasteryzacji koktajli na podstawie danych o składnikach, kategoriach i typach szklanek.

## Wymagania
Aby uruchomić projekt, potrzebujesz Pythona w wersji 3.8 lub nowszej oraz zainstalowanych odpowiednich bibliotek.

## Instalacja

### Zainstaluj zależności z pliku `dependencies.txt`
Aby zainstalować wszystkie wymagane biblioteki, użyj poniższej komendy:
```bash
pip install -r dependencies.txt
```
## Uruchamianie

1. **Analiza statystyk opisowych**:
   Aby obliczyć i wyświetlić statystyki opisowe dla liczby składników, uruchom skrypt `eda.py`:
   ```bash
   python src/eda.py
   ```

2. **Przygotowanie danych**:
   Aby wczytać i przetworzyć dane, uruchom skrypt `preprocessing.py`:
   ```bash
   python src/preprocessing.py
   ```

3. **Klasteryzacja**:
   Po przetworzeniu danych, uruchom skrypt `clustering.py`, aby przeprowadzić klasteryzację:
   ```bash
   python src/clustering.py
   ```

## Wyniki
Wyniki klasteryzacji zostaną zapisane w katalogu `figures`, a przetworzone dane w katalogu `data`