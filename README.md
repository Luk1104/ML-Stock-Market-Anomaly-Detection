# Detekcja anomalii w danych giełdowych
## Stock Market Anomaly Detection via Yahoo Finance

### Cel projektu
Wykrywanie anomalii cenowych w notowaniach giełdowych — gwałtownych skoków lub spadków kursów, które mogą wskazywać na manipulację rynkiem, cyberatak na infrastrukturę giełdy lub insider trading.

### Dane
Dane pobierane automatycznie przez bibliotekę yfinance — historyczne notowania wybranych spółek (np. AAPL, MSFT, TSLA, GPW: PKN, CDR) z ostatnich 5 lat. Etykiety anomalii generowane automatycznie (np. zmiana > 3 odchylenia standardowe).

### Eksperyment 1
Ekstrakcja i porównanie zestawów cech: (a) surowe zwroty dzienne, (b) rolling statistics (średnia krocząca 5/20 dni, odchylenie std), (c) wskaźniki techniczne (RSI, Bollinger Bands) — który zestaw lepiej separuje anomalie?

### Eksperyment 2
Klasyfikacja niezbalansowana: anomalie stanowią ~2-5% danych. Porównanie wyników bez resamplingu vs. z SMOTE — metryki F1 i precision dla klasy anomalii.

### Analiza statystyczna
Wizualizacja wykrytych anomalii na wykresie cenowym (matplotlib), test Wilcoxona dla porównania konfiguracji w CV.

### Artefakt
Skrypt .py pobierający dane przez yfinance, obliczający cechy, klasyfikujący anomalie i rysujący wykres z oznaczonymi anomaliami.

### Biblioteki
numpy, scikit-learn, imbalanced-learn, matplotlib, scipy, yfinance (pip install yfinance)

## Wskazówki dla studentów:
- yfinance: ticker.history(period='5y') pobiera dane — prosto i szybko.
- Etykiety anomalii: zscore > 3 lub zmiana dzienna > 2*std jako punkt startowy.
- Warto pokazać kilka spółek i porównać ich 'anomalność' na jednym wykresie.
- Projekt ma świetny efekt wizualny — wykres z zaznaczonymi anomaliami robi wrażenie na prezentacji.