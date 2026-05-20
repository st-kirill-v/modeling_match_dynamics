# Modeling Match Dynamics

Проектная версия лабораторной работы `task2_lstm_match_dynamics_fixed.ipynb`.

Пайплайн сохранён по смыслу:

- Football Events переводится из event-level в minute-level.
- Основные football targets: `home_scores_next_half`, `away_scores_next_half`.
- Football LSTM видит только первый тайм (`time <= 45`) и предсказывает голы во втором тайме.
- Старые short-horizon football targets не используются.
- NBA Movement Data используется как possession-level proxy task с target `possession_is_dangerous`.
- NBA proxy не выдаётся за настоящий прогноз счёта.
- Для честного NBA scoring prediction нужен join с play-by-play.

## Запуск через uv

Установить зависимости:

```powershell
uv sync --python 3.13
```

Быстрый локальный запуск без скачивания NBA:

```powershell
uv run python scripts\run_pipeline.py --football-path "D:\Учеба\Глубокое обучение (DL)\Football Events.zip" --skip-nba-download
```

Быстрая проверка архитектуры без обучения LSTM:

```powershell
uv run python scripts\run_pipeline.py --football-path "D:\Учеба\Глубокое обучение (DL)\Football Events.zip" --skip-lstm --skip-nba-download
```

Если NBA JSON уже распакованы локально:

```powershell
uv run python scripts\run_pipeline.py --football-path "D:\Учеба\Глубокое обучение (DL)\Football Events.zip" --nba-json-dir "path\to\nba_json"
```

## NBA Matched Dataset

Чтобы собрать локальный NBA датасет из 50 матчей без полного `git clone`:

```powershell
python scripts\build_nba_matched_dataset.py --max-games 50 --moment-stride 50
```

Результат:

```text
data/nba_matched/nba_matched_events_50.csv
```

Одна строка в этом CSV - `game_id + event_id`. Признаки берутся из movement JSON, а реальные labels берутся из `events/*.csv` и `shots.csv`:

- `shot_attempt`
- `shot_made`
- `shot_missed`
- `free_throw`
- `turnover`
- `foul`
- `scoring_event`
- `has_shot_chart_row`

## Структура

- `src/match_dynamics/config.py` - настройки, пути, targets и списки признаков.
- `src/match_dynamics/data_loading.py` - загрузка Football Events и NBA Movement Data.
- `src/match_dynamics/football.py` - football preprocessing, proxy-xG, rolling/momentum/pressure/team strength.
- `src/match_dynamics/nba.py` - NBA tracking parsing и possession-level proxy target.
- `src/match_dynamics/sequences.py` - split, scaling, LSTM sequence generation.
- `src/match_dynamics/models.py` - LSTM и baseline-модели.
- `src/match_dynamics/evaluation.py` - метрики: PR-AUC, ROC-AUC, Brier, calibration, top-decile lift.
- `src/match_dynamics/visualization.py` - графики из ноутбука.
- `src/match_dynamics/pipeline.py` - orchestration всего пайплайна.
- `scripts/run_pipeline.py` - CLI-точка запуска.

## GitHub

В репозиторий нужно коммитить исходный код, `pyproject.toml`, `uv.lock`, `.python-version`, `.env.example`, `.gitignore`, `.gitattributes` и README.

Не нужно коммитить:

- `.venv/`
- `data/`
- реальные `.env`
- `kaggle.json`
- результаты в `outputs/figures`, `outputs/metrics`, `outputs/models`

## Важное ограничение Colab

Colab не имеет доступа к локальному диску `D:\`. Для Colab нужно загрузить `Football Events.zip` вручную, положить его рядом с ноутбуком/скриптом или подключить Google Drive.
