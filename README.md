# SmartBuyer Embedding API

FastAPI-сервис для генерации OpenAI эмбеддингов. Используется для семантического поиска по профилям.

## 🚀 Установка

1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/sammywo0w/smartbuyer_pyton.git
   cd smartbuyer_pyton
Установить зависимости:

bash
Копировать
Редактировать
pip install -r requirements.txt
Создать .env и добавить OpenAI API ключ:

bash
Копировать
Редактировать
OPENAI_API_KEY=sk-...
🧪 Запуск
bash
Копировать
Редактировать
uvicorn main:app --reload
📬 Использование
POST /embed

Тело запроса:

json
Копировать
Редактировать
{
  "text": "Ищу специалиста по закупкам"
}
Ответ:

json
Копировать
Редактировать
{
  "embedding": [0.012, -0.034, ...]
}
Копировать
Редактировать
