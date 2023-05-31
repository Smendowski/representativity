## 1. Wprowadzenie

<p style="text-align: justify;">
Serwis reprezentatywności umożliwia trening i wykorzystanie modeli uczenia maszynowego do prognozowania
reprezentatywności obiektów. Wykorzystuje on metrykę opartą na analizie K najbliższych sąsiadów w danym fragmencie
zbioru danych. Dane wejściowe są dzielone losowo na L fragmentów, a każdy z nich przetwarzany jest współbieżnie
a następnie wykorzystywany przez niezależne modele. Ostateczny wynik reprezentatywności jest obliczany jako średnia
wartość prognoz z uprzednio przygotowanych modeli składowych. W konsekwencji, zadany problem zrealizowano za pomocą 
regresji nadzorowanej z wykorzystaniem Lasów Losowych.
</p>

## 2. Konfiguracja środowiska
1. Utworzenie pliku `.env` w głównym katalogu projektu oraz zdefiniowanie zmiennych środowiskowych zgodnie 
z poniższym przykładem.
```shell
  NUMBER_OF_ENSEMBLE_MODELS = 5
  N_NEIGHBORS = 5
```
2. Docker - weryfikacja oprogramowania
```shell
  docker --version
  >> Docker version 20.10.17, build 100c701
```
3. Zbudowanie obrazu i uruchomienie serwisu reprezentatywności
```shell
  docker build -t representativeness-service .
  docker run --rm -p 9000:8000 representativeness-service

  [...]
  INFO:     Started server process [1]
  INFO:     Waiting for application startup.
  INFO:     Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
  
  
  Alternatywnie, pobranie obrazu z DockerHub:
  docker pull smendowski/representativeness-service:latest
  docker run --rm -p 9000:8000 smendowski/representativeness-service
```
4. Zbudowanie obrazu i uruchomienie testów jednostkowych
```shell
  docker build -t representativeness-service-tests  --target tester .
  docker run --rm representativeness-service-tests pytest
  
  Alternatywnie, pobranie obrazu z DockerHub:
  docker pull smendowski/representativeness-service-tests:latest
  docker run --rm smendowski/representativeness-service-tests pytest
```

## 3. Scenariusze eksperymentalne
<p style="text-align: justify;">
Przykładowe scenariusze eksperymentalne zdefiniowano w oparciu o trzy uprzednio utworzone zbiory danych, zamieszczone
w katalogu `artifacts`, które różnią się liczbą próbek a także wymiarowością cech próbek w poszczególnych zbiorach.
Szczególną rolą zbioru danych L jest możliwość weryfikacji responsywności serwisu podczas treningu modelu.
</p>

```shell
Zbiór danych S: dataset_1_000_samples_5_features.json
Zbiór danych M: dataset_10_000_samples_10_features.json
Zbiór danych L: dataset_100_000_samples_10_features.json
```

### 3.1 Wykorzystanie zbioru danych S
##### Krok 1. Weryfikacja endpointu *POST /predict*
```shell
curl -X POST \
     -H "Content-Type: application/json" \
     -d @artifacts/samples_5_features.json \
     http://127.0.0.1:9000/predict
```
```json
{
  "detail": "Prediction cannot be made. Regressor is not fitted yet"
}
```
<p style="text-align: justify;">
Prognozowanie nie jest możliwe jeśli nie zlecono treningu modelu na zadanym zbiorze danych.
</p>


##### Krok 2. Weryfikacja endpointu *GET /status*
```shell
curl -X GET http://127.0.0.1:9000/status
```
```json
{
  "status": "Training has not started yet"
}
```
##### Krok 3. Rozpoczęcie treningu *POST /train* z wykorzystaniem zbioru danych S
```shell
curl -X POST \
     -H "Content-Type: application/json" \
     -d @artifacts/dataset_1_000_samples_5_features.json \
     http://127.0.0.1:9000/train
```
```json
{
  "detail": "Job has been submitted"
}
```
##### Krok 4. Weryfikacja endpointu *GET /status*
```shell
curl -X GET http://127.0.0.1:9000/status
```
```json
{
  "status": "Training has finished",
  "start_time":"2023-05-30 19:56:12",
  "finish_time":"2023-05-30 19:56:12"
}
```

##### Krok 5. Weryfikacja endpointu *POST /predict*
```shell
curl -X POST \
     -H "Content-Type: application/json" \
     -d @artifacts/samples_5_features.json \
     http://127.0.0.1:9000/predict
```
```json
{
  "representativeness": [
    0.72519,
    0.7738,
    0.7497,
    0.74954,
    0.7609
  ]
}
```

### 3.2 Wykorzystanie zbioru danych M
##### Krok 1. Rozpoczęcie treningu *POST /train* z wykorzystaniem zbioru danych M
```shell
curl -X POST \
     -H "Content-Type: application/json" \
     -d @artifacts/dataset_10_000_samples_10_features.json \
     http://127.0.0.1:9000/train
```
```json
{
  "detail": "Job has been submitted"
}
```

##### Krok 2. Weryfikacja endpointu *GET /status*
```shell
curl -X GET http://127.0.0.1:9000/status
```
```json
{
  "status": "Training in progress",
  "start_time":"2023-05-30 20:06:45"
}
```

##### Krok 3. Weryfikacja endpointu *GET /status*
```shell
curl -X GET http://127.0.0.1:9000/status
```
```json
{
  "status": "Training has finished",
  "start_time": "2023-05-30 20:06:45",
  "finish_time": "2023-05-30 20:06:49"
}
```

##### Krok 4. Weryfikacja endpointu *POST /predict* - lista próbek o nieprawidłowej wymiarowości
```shell
curl -X POST \
     -H "Content-Type: application/json" \
     -d @artifacts/samples_5_features.json \
     http://127.0.0.1:9000/predict
```
```json
{
  "detail": "Inference sample has unexpected shape: (5,). Expected shape: (10,)"
} 
```

##### Krok 5. Weryfikacja endpointu *POST /predict* - lista próbek o prawidłowej wymiarowości
```shell
curl -X POST \
     -H "Content-Type: application/json" \
     -d @artifacts/samples_10_features.json \
     http://127.0.0.1:9000/predict
```
```json
{
  "representativeness": [
    0.64725,
    0.66258,
    0.6528,
    0.63925,
    0.64187
  ]
}
```


### 3.3 Wykorzystanie zbioru danych L
##### Krok 1. Rozpoczęcie treningu *POST /train* z wykorzystaniem zbioru danych L
```shell
curl -X POST \
     -H "Content-Type: application/json" \
     -d @artifacts/dataset_100_000_samples_10_features.json \
     http://127.0.0.1:9000/train
```
```json
{
  "detail": "Job has been submitted"
}
```
##### Krok 2. Weryfikacja endpointu *GET /status*
```shell
 curl -X GET http://127.0.0.1:9000/status
```
```json
{
  "status": "Training in progress",
  "start_time":"2023-05-30 20:11:10"
}
```
Poprzez wykorzystanie zbioru danych L możliwe jest zaobserwowanie znacznie dłuższego trenowania modeli.

##### Krok 3. Weryfikacja endpointu *GET /status*
```shell
 curl -X GET http://127.0.0.1:9000/status
```
```json
{
  "status": "Training has finished",
  "start_time": "2023-05-30 20:11:10",
  "finish_time": "2023-05-30 20:11:52"
}
```
##### Krok 4. Weryfikacja endpointu *POST /predict*
```shell
curl -X POST \
     -H "Content-Type: application/json" \
     -d @artifacts/samples_10_features.json \
     http://127.0.0.1:9000/predict
```
```json
{
  "representativeness": [
    0.69789,
    0.71712,
    0.71213,
    0.69799,
    0.70224
  ]
}
```

## 4. Wykorzystane technologie
FastAPI, Asyncio, Pydantic, PyTest, Docker multi-stage build, GitHub Actions.
