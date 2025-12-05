Безбашенное функциональное программирование на Python

# Введение
Функциональное программирование в Python — это искусство балансирования между верностью парадигме и практическими ограничениями языка.

В этой статье основной акцент сделан на аспектах функционального программирования на Python:
- Почему Python не очень подходит для функционального программирования
- Почему уметь писать функциональный код даже на таком языке ХОРОШО 
- Конструирование функциональных преобразований без условных операторов и циклов
- Использовании встроенных функций и методов классов как параметров
- Почему генераторы являются функциональными конструктами, но в функциональной парадигме им места нет

Статья написана QWEN, с моих тезисов, которые были сформулированы мной во время кодирования на питоне в функциональном стиле (для универского проекта). Примеры абстрактные, да и вряд ли на 100% рабочие — для меня было главным передать идею.

# Почему Python не подходит для функционального программирования с точки зрения производительности и идиоматики, и почему всё равно так хочется использовать функциональность
## Производительность: интерпретатор против функциональной парадигмы

Python — интерпретируемый язык с динамической типизацией, и его архитектура изначально не заточена под функциональное программирование. Давайте разберём ключевые проблемы производительности:

### 1. Накладные расходы на вызов функций
В CPython каждый вызов функции требует создания нового фрейма стека, проверки аргументов и управления контекстом. Это критично для функционального стиля, где используются множественные вызовы маленьких функций (например, в `map` или `filter`).

**Пример сравнения скорости:**
```python
import timeit

# Тест 1: map с лямбдой
map_test = "list(map(lambda x: x*2, range(1000000)))"
# Тест 2: генератор списка
gen_test = "[x*2 for x in range(1000000)]"
# Тест 3: цикл for
loop_test = """
result = []
for x in range(1000000):
    result.append(x*2)
"""

print("map:", timeit.timeit(map_test, number=10))
print("gen:", timeit.timeit(gen_test, number=10))
print("loop:", timeit.timeit(loop_test, number=10))
```

**Результат:**
```
map: 1.5791718600085005
gen: 0.9516333599895006
loop: 1.1159681400022237
```

Почему так происходит? При компиляции генераторов в байт-код CPython применяет специальные оптимизации (например, `LIST_APPEND` вместо вызова `.append()`), тогда как `map` требует постоянного переключения контекста между Python-функцией и C-реализацией.

### 2. Отсутствие tail call optimization
В функциональных языках рекурсивные вызовы оптимизируются через TCO (Tail Call Optimization), но в Python:

```python
def factorial(n, acc=1):
    return acc if n == 0 else factorial(n-1, acc*n)

# Упадёт при n > 999 из-за переполнения стека
factorial(1000)  # RecursionError
```

Интерпретатор не оптимизирует хвостовую рекурсию, что делает рекурсивные алгоритмы непрактичными для больших данных.

### 3. Глобальная блокировка интерпретатора (GIL)
Даже при попытке распараллелить функциональные операции через `multiprocessing` возникают накладные расходы на сериализацию данных между процессами, что часто сводит на нет выгоду от параллелизма.


## Почему функциональный подход имеет место быть

Несмотря на эти ограничения, функциональный подход имеет смысл в следующих случаях:

### 1. Читаемость потоков данных и пайплайнов
При работе с данными цепочки преобразований иногда естественнее выражать в декларативном стиле.

**Императивный стиль:**
```python
def process_transactions(transactions: List[Transaction], min_amount: float) -> List[Report]:
    results = []
    for tx in transactions:
        if tx.amount >= min_amount and tx.status == "completed":
            report = Report(
                tx_id=tx.id,
                amount=tx.amount,
                currency=tx.currency,
                category="high_value" if tx.amount > 1000 else "standard"
            )
            results.append(report)
    return results
```

**Функциональный пайплайн (retun читается с конца к началу):**
```python
def process_transactions(transactions: List[Transaction], min_amount: float) -> List[Report]:
    def is_valid(tx: Transaction) -> bool:
        return tx.amount >= min_amount and tx.status == "completed"
    
    def create_report(tx: Transaction) -> Report:
        return Report(
            tx_id=tx.id,
            amount=tx.amount,
            currency=tx.currency,
            category="high_value" if tx.amount > 1000 else "standard"
        )
    
    return \
        list(
            map(create_report,
                filter(is_valid,
                       transactions
                )))
```

Обратите внимание: функциональный вариант читается **справа налево**, что соответствует логике обработки данных — мы видим полную цепочку преобразований в одном выражении. Это особенно ценно при отладке пайплайнов (например, в ETL-процессах). 
### 2. Отсутствие побочных эффектов
Функции без состояния проще тестировать и параллелить. Даже в Python можно создавать "чистые" функции:

```python
def process_user(user: dict) -> dict:
    """Чистая функция обработки данных"""
    return {
        **user,
        "full_name": f"{user['first_name']} {user['last_name']}",
        "age_group": "adult" if user["age"] >= 18 else "minor"
    }

# Тестирование не требует настройки контекста
assert process_user({"first_name": "John", "last_name": "Doe", "age": 25}) == {
    "first_name": "John",
    "last_name": "Doe",
    "age": 25,
    "full_name": "John Doe",
    "age_group": "adult"
}
```


**Пример из реального ETL:**
```python
def clean_data(raw):
    """
    Очищает данные: фильтрует активные записи, добавляет score,
    сортирует по убыванию score и возвращает список.
    """

    return \
		    list(
		        sorted(
		            map(
		                lambda x: {**x, "score": calculate_score(x)},
		                filter(
		                    lambda x: x["status"] == "active",
		                    raw
		                )
		            ),
		            key=lambda x: x["score"],
		            reverse=True
		        )
		    )
```

### 3. Удобная параллелизация при выключенном GIL
Функциональные операции легко параллелить, что наверняка даст прирост к производительности на Python 3.14 с выключенной блокировкой интерпретатора.

## Когда стоит использовать функциональный подход в Python

1. **Обработка данных**: ETL-процессы, аналитика, машинное обучение
2. **Конфигурация пайплайнов**: когда логика представляет собой последовательность преобразований
3. **Параллельные вычисления**: через `concurrent.futures` с чистыми функциями
4. **Тестирование**: изолированные функции проще покрываются юнит-тестами

**Важно:** Не стоит насиловать Python в попытках сделать его Haskell'ом. Используйте функциональные паттерны там, где они дают преимущество в читаемости и поддерживаемости, но помните о компромиссах в производительности. Для критичных к скорости участков кода лучше придерживаться генераторов и встроенных методов, а не классических `map`/`filter`.

# Лямбды

## Лямбды: мощь и ограничения

Лямбды в Python — это анонимные функции, которые позволяют определять короткие операции прямо в месте использования. Их синтаксис лаконичен:

```python
double = lambda x: x * 2
print(double(5))  # 10
```

Однако у лямбд есть серьёзные ограничения:
- Только одно выражение в теле (нельзя использовать условные операторы, операторы цикла)
- Отсутствие документации и имени (сложнее отлаживать)
- Меньшая производительность по сравнению со встроенными функциями
- Проблемы с типизацией (аннотации типов в лямбдах выглядят громоздко)

**Пример проблемной лямбды:**
```python
# Неочевидный и трудный для отладки код
process = lambda x: (
    x * 2 if x > 0 else 
    abs(x) // 2 if x < -10 else 
    0
)
```

# Стандартные функции и методы классов вместо передаваемых функций

## Почему стандартные функции часто лучше лямбд

### 1. Читаемость и идиоматичность
Встроенные функции и методы классов делают код самодокументируемым. Сравните:

```python
input_data = ["  Apple  ", "BANANA", "  cherry  ", "  ", "Date", ""]

# Функциональная обработка
cleaned = \
			list(
			    map(str.upper,
			        sorted( 
				        map(str.strip, input_data),
				        key=str.lower
					)
			    )
			)
print(cleaned)

# Иперативный код
cleaned = []
for item in input_data:
    stripped = item.strip()
    if stripped:  # фильтрация пустых строк
        cleaned.append(stripped.upper())
cleaned.sort(key=str.lower)
print(cleaned)


# ['APPLE', 'BANANA', 'CHERRY', 'DATE']
```

### 2. Прямой доступ к методам классов
Многие методы можно передавать напрямую без обёртки в лямбду:

```python
# Замена лямбды str.lower()
words = ["Apple", "Banana", "cherry"]
sorted(words, key=lambda s: s.lower())  # Избыточно
sorted(words, key=str.lower)           # Идиоматично

# Замена len()
names = ["Alice", "Bob", "Charles"]
sorted(names, key=lambda x: len(x))    # Не нужно
sorted(names, key=len)                 # Прямо и понятно
```

### 3. Магические методы как функции
Даже специальные методы (`__len__`, `__add__`) можно использовать напрямую:

```python
# С лямбдой
total = sum(map(lambda x: x.__len__(), ["apple", "banana", "cherry"]))

# С магическим методом
total = sum(map(str.__len__, ["apple", "banana", "cherry"]))

# Самый простой вариант
total = sum(map(len, ["apple", "banana", "cherry"]))
```

**Важно:** `str.__len__` работает только для строк, тогда как `len` универсален благодаря протоколу последовательностей в Python.

### 4. Модуль operator — лучший друг
Этот стандартный модуль предоставляет функции для всех базовых операций:

```python
from operator import itemgetter, attrgetter, methodcaller

# Получение элемента по индексу
data = [("apple", 3), ("banana", 1), ("cherry", 5)]
sorted(data, key=itemgetter(1))  # Сортировка по второму элементу

# Получение атрибута объекта
class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age

users = [User("Alice", 30), User("Bob", 25)]
sorted(users, key=attrgetter("age"))

# Вызов метода
texts = ["  hello  ", "  world  "]
list(map(methodcaller("strip"), texts))  # ["hello", "world"]
```

## Производительность: лямбды vs стандартные функции

Проведём замеры скорости для типичной операции:
```python
import timeit
from operator import itemgetter

data = [(i, i*2) for i in range(10000)]

# Тест 1: лямбда
lambda_time = timeit.timeit(
    "sorted(data, key=lambda x: x[1])", 
    globals=globals(), 
    number=100
)

# Тест 2: itemgetter
itemgetter_time = timeit.timeit(
    "sorted(data, key=itemgetter(1))", 
    globals=globals(), 
    number=100
)

print(f"Лямбда: {lambda_time:.4f}s")
print(f"Itemgetter: {itemgetter_time:.4f}s")
```

**Типичный результат:**
```
Лямбда: 0.0929s
Itemgetter: 0.0285s
```

**Почему так происходит?**
1. Лямбды — это полноценные функции Python с накладными расходами на вызов
2. `itemgetter` реализован на C и оптимизирован для быстрого доступа
3. Интерпретатору проще оптимизировать вызовы встроенных функций

## Практические рекомендации

1. **Избегайте лямбд для всего, что имеет имя в стандартной библиотеке**
   ```python
   # Плохо
   list(map(lambda x: x.strip(), texts))
   
   # Хорошо
   list(map(str.strip, texts))
   ```

2. **Используйте operator вместо математических лямбд**
   ```python
   from operator import mul
   
   # Плохо
   list(map(lambda x, y: x * y, [1,2,3], [4,5,6]))
   
   # Хорошо
   list(map(mul, [1,2,3], [4,5,6]))
   ```

3. **Для сложных операций пишите именованные функции**
   ```python
   # Плохо (многострочная лямбда через tuple)
   process = lambda x: (
       print(f"Start: {x}"),
       x * 2,
       print(f"End: {x*2}")
   )[-2]
   
   # Хорошо
   def process(x):
       """Удваивает значение с логированием"""
       print(f"Start: {x}")
       result = x * 2
       print(f"End: {result}")
       return result
   ```

4. **Для сортировки всегда проверяйте наличие готового решения**
   ```python
   # Для сортировки по нескольким полям
   sorted(users, key=lambda u: (u.age, u.name))
   
   # Или через itemgetter
   from operator import itemgetter
   sorted(users, key=itemgetter("age", "name"))
   ```

**Золотое правило:** Если ваша лямбда занимает больше одной строки или содержит вложенные условия — пора выносить её в именованную функцию. Стандартные функции и методы классов не только ускоряют код, но и делают его понятным для других разработчиков, которые сразу узнают знакомые паттерны.


# Функции reduce, map, filter, filterfalse, all, any, starmap и почему при функциональном программировании  нужно использовать их а не генераторы

## Почему генераторы "ломают" функциональный стиль

Генераторы в Python — мощный инструмент, но они создают иллюзию императивного кода внутри функциональной парадигмы. Рассмотрим ключевую проблему:

```python
# Генератор
result = [x*2 for x in range(10) if x % 2 == 0]

# Функциональный стиль
result = list(map(lambda x: x*2, filter(lambda x: x % 2 == 0, range(10))))
```

**Проблема читаемости порядка операций:**
- В генераторе мы читаем слева-направо: *сначала преобразование, потом условие, потом источник*
- В функциональном стиле порядок соответствует логике обработки: *сначала источник → фильтр → преобразование*

Это критично при построении сложных пайплайнов. Сравните:

```python
# Генератор (запутанный порядок)
cleaned = [
    normalize(item) 
    for item in load_data() 
    if is_valid(item) 
    for item in preprocess(item)
]

# Функциональный пайплайн (естественный порядок)
from itertools import starmap

  

cleaned = \
			list(
			    map(
			        normalize,
			        starmap(
			            preprocess,
			            filter(
			                is_valid,
				                load_data()
			            )
			        )
			    )
			)
```

Генераторы нарушают принцип "конвейерной обработки", заставляя читать код "внутри наружу", что противоречит философии функционального программирования.

## Порядок чтения: генераторы vs вложенные функции

**Генераторы читаются СЛЕВА НАПРАВО (как естественный порядок выполнения):**
```python
cleaned = [
    normalize(item["value"])  # 4. Последнее преобразование
    for item in load_data()   # 1. Источник данных
    if item["status"] == "active"  # 2. Фильтрация
    for item in preprocess(item["raw"])  # 3. Промежуточная обработка
]
```

**Вложенные функции читаются СПРАВА НАЛЕВО (как математическая композиция):**
```python
cleaned = list(
    map(normalize,  # 4. Последнее преобразование
        filter(lambda x: x["status"] == "active",  # 2. Фильтрация
            starmap(preprocess,  # 3. Промежуточная обработка
                map(itemgetter("raw"),  # 1. Источник данных
                    load_data()
                )
            )
        )
    )
)
```

Реальный пример

Давайте возьмём простой пример обработки данных:

```python
# Генератор (читается в порядке выполнения)
result = [
    x * 2  # 3. Удваиваем
    for x in range(10)  # 1. Источник
    if x > 5  # 2. Фильтрация
]

# Вложенные функции (читается в обратном порядке)
result = \
list(
    map(lambda x: x * 2,  # 3. Удваиваем
        filter(lambda x: x > 5,  # 2. Фильтрация
            range(10)  # 1. Источник
        )
    )
)
```

**Порядок выполнения в обоих случаях одинаков:**
1. `range(10)` → генерируем числа от 0 до 9
2. `x > 5` → оставляем только числа 6,7,8,9
3. `x * 2` → умножаем на 2, получаем [12,14,16,18]

**Но порядок чтения разный:**
- В генераторе мы читаем код в том же порядке, в котором выполняются операции
- Во вложенных функциях мы сначала видим конечное преобразование, и только потом доходим до источника данных

## Почему это важно для ETL

В ETL-процессах (Extract, Transform, Load) порядок операций критичен. Сравните:

**Генератор (естественный порядок чтения):**
```python
processed = [
    normalize_sales(sale)  # 4. Нормализация продаж
    for sale in load_sales_data()  # 1. Загрузка данных
    if sale["region"] == "EU"  # 2. Фильтрация по региону
    for sale in clean_currency(sale)  # 3. Очистка валюты
]
```

**Вложенные функции (обратный порядок чтения):**
```python
processed = \
			list(
			    map(normalize_sales,  # 4. Нормализация продаж
			        starmap(clean_currency,  # 3. Очистка валюты
			            filter(lambda s: s["region"] == "EU",  # 2. Фильтрация по региону
			                load_sales_data()  # 1. Загрузка данных
			            )
			        )
			    )
			)
```

Генераторы действительно проще читаются для человека, так как порядок записи соответствует порядку выполнения операций. Это делает их более интуитивными для понимания последовательности обработки данных.

## Когда использовать каждый стиль

**Используйте генераторы, когда:**
- Нужна максимальная читаемость и соответствие порядку выполнения
- Есть вложенные циклы или несколько условий
- Код будет часто модифицироваться другими разработчиками

**Используйте вложенные функции, когда:**
- Работаете с библиотеками, которые используют функциональный стиль (Pandas, PySpark)
- Нужно передавать цепочку преобразований как объект
- Хотите подчеркнуть математическую композицию функций
## Изоляция побочных эффектов
Функции высшего порядка поощряют создание чистых функций:

```python
# Генератор с побочным эффектом (плохо!)
results = [print(x) or x*2 for x in range(5)]

# Функциональный подход (чистые функции)
list(map(lambda x: (print(x), x*2)[1], range(5)))  # Все еще плохо

# Правильный подход
def process(x):
    print(f"Processing {x}")
    return x*2

list(map(process, range(5)))
```

## Легкая композиция
Функции высшего порядка идеально работают с библиотеками вроде `toolz` или `funcy`:

```python
from toolz import compose, pipe

process = compose(
    partial(map, normalize),
    partial(filter, is_valid),
    partial(starmap, preprocess)
)

result = pipe(
    load_data(),
    process,
    list
)
```


## `map` — чистое преобразование без побочных эффектов

**Почему лучше генератора:**
- Явно декларирует операцию преобразования
- Сохраняет ленивость (в Python 3 возвращает iterator)
- Лучше работает в композиции функций
- Идиоматична для функционального программирования

**Производительность:**
```python
import timeit

# Тест 1: генератор
gen_time = timeit.timeit("[x**2 for x in range(10000)]", number=1000)

# Тест 2: map
map_time = timeit.timeit("list(map(lambda x: x**2, range(10000)))", number=1000)

print(f"Генератор: {gen_time:.4f}s")
print(f"Map: {map_time:.4f}s")
```
Результаты показывают, что генераторы обычно быстрее, но разница минимальна (5-10%). Выгода в читаемости пайплайнов перевешивает эту разницу.


## `filter` и `filterfalse` — декларативная фильтрация

**Ключевое преимущество:** разделение логики фильтрации и преобразования.

```python
from itertools import filterfalse

# Генератор с двойным условием
valid = [x for x in data if x > 0 if x < 100]

# Функциональный стиль
valid = filter(lambda x: 0 < x < 100, data)

# Обратная фильтрация через filterfalse
invalid = filterfalse(lambda x: 0 < x < 100, data)
```

**Реальный пример обработки логов:**
```python
# Генератор (многоуровневые условия)
errors = [
    parse_log(line) 
    for line in logs 
    if "ERROR" in line 
    if not is_ignored(line)
]

# Функциональный подход
from itertools import filterfalse


errors =\
		 list(
		    map(
		        parse_log,
		        filterfalse(
		            is_ignored,
			            filter(
			                lambda l: "ERROR" in l,
			                logs
			            ))))
```

Здесь `filterfalse` из `itertools` идеально заменяет отрицательные условия, делая код самодокументируемым.

---

## `reduce` — мощная агрегация данных

**Почему не использовать цикл for:**
- `reduce` явно выражает операцию свертки
- Соответствует математической нотации
- Лучше читается в композиции

```python
from functools import reduce

# Цикл for (императивный)
total = 0
for num in [1, 2, 3, 4]:
    total += num

# Reduce (декларативный)
total = reduce(lambda acc, x: acc + x, [1, 2, 3, 4], 0)

# С operator.add (еще читабельнее)
from operator import add
total = reduce(add, [1, 2, 3, 4], 0)
```

**Сложные агрегации:**
```python
# Группировка данных
data = [("a", 1), ("b", 2), ("a", 3)]
grouped = reduce(
    lambda acc, pair: {**acc, pair[0]: acc.get(pair[0], []) + [pair[1]]},
    data,
    {}
)
# {'a': [1, 3], 'b': [2]}

# Эквивалент через цикл (менее читаемо)
grouped = {}
for key, value in data:
    if key not in grouped:
        grouped[key] = []
    grouped[key].append(value)
```


## `all` и `any` — семантически правильные проверки

**Преимущество перед генераторами:**
- Немедленная остановка при достижении результата
- Явное выражение намерения (проверка всех/хотя бы одного)

```python
# Генератор с any()
if any(x < 0 for x in numbers):
    print("Есть отрицательные")

# Прямой вызов any()
if any(map(lambda x: x < 0, numbers)):
    print("Есть отрицательные")

# Проверка всех элементов
if all(map(str.isalpha, words)):
    print("Все слова содержат только буквы")
```

**Важный нюанс:** `all` и `any` работают лениво — останавливаются при первом ложном/истинном результате. Это критично для потоковых данных.


## `starmap` — обработка аргументов с распаковкой

**Уникальная возможность:** автоматическая распаковка кортежей в аргументы функции.

```python
from itertools import starmap

# Генератор с распаковкой
results = [pow(x, y) for x, y in [(2,3), (3,2), (4,2)]]

# starmap (чище и идиоматичнее)
results = starmap(pow, [(2,3), (3,2), (4,2)])

# Реальный пример: обработка координат
points = [(1, 2), (3, 4), (5, 6)]
distances = starmap(
    lambda x, y: (x**2 + y**2) ** 0.5,
    points
)
```

**Сравнение с map:**
```python
# С map потребуется вложенная лямбда
distances = map(lambda p: (p[0]**2 + p[1]**2) ** 0.5, points)

# starmap делает это явно
distances = starmap(lambda x, y: (x**2 + y**2) ** 0.5, points)
```


## Практические рекомендации

1. **Используйте операторную форму там, где возможно**
   ```python
   # Вместо лямбд
   from operator import add, methodcaller
   
   total = reduce(add, numbers)
   stripped = map(methodcaller("strip"), texts)
   ```

2. **Комбинируйте с itertools для сложных операций**
   ```python
   from itertools import takewhile, dropwhile
   
   # Обработка данных до первого невалидного элемента
   valid = list(takewhile(is_valid, data))
   ```

4. **Избегайте смешивания стилей в одном выражении**
   ```python
   # Плохо (гибридный стиль)
   result = [x*2 for x in filter(lambda y: y > 0, data)]
   
   # Хорошо (чистый функциональный)
   result = map(lambda x: x*2, filter(lambda x: x > 0, data))
   
   # Или чистый генератор
   result = (x*2 for x in data if x > 0)
   ```

# Условные выражения. if-else в одну строку. match/case

## Архитектура тернарного оператора: не просто синтаксический сахар

Тернарный оператор в Python имеет особую структуру: `value_if_true if condition else value_if_false`. Это не просто компактная запись, а принципиально другой способ организации логики:

### 1. Ленивые вычисления
Только одно из выражений вычисляется, в зависимости от условия:
```python
# x() не вызывается, если condition == False
result = x() if condition else y()
```

### 2. Композируемость
Тернарные операторы можно вкладывать и комбинировать с другими функциональными конструкциями:

```python
# Вложенная логика без разрушения пайплайна
classify = lambda x: (
    "positive" if x > 0 else 
    "negative" if x < 0 else 
    "zero"
)

# Интеграция с map
results = map(
    lambda x: x**2 if x > 0 else abs(x) if x < -10 else 0,
    [-15, -5, 0, 5, 15]
)
```

### 3. Типовая согласованность
В отличие от обычных `if`, тернарный оператор гарантирует возврат значения всегда одного типа (если логика корректна), что критично для статической типизации:

```python
from typing import Literal

def sign(x: float) -> Literal[-1, 0, 1]:
    return -1 if x < 0 else 1 if x > 0 else 0
```


## Паттерны использования в функциональном стиле

### 1. Обработка ошибок без исключений
```python
import math
# Вместо try/except в пайплайне

safe_sqrt = lambda x: x**0.5 if x >= 0 else float('nan')

results = filter(
    lambda x: not math.isnan(x),
    map(safe_sqrt, data)
)
```

### 2. Условные преобразования
```python
# Нормализация данных с разными стратегиями
normalize = lambda x: (
    x / max_value if x > threshold 
    else x * 2 if x < 0 
    else x
)

# В пайплайне обработки
processed = map(
    lambda x: x.upper() if x.islower() else x.lower() if x.isupper() else x,
    ["HELLO", "world", "MiXeD"]
)
```

### 3. Селекция функций
```python
# Выбор алгоритма в зависимости от условия
process = lambda data, mode: (
    fast_process(data) if mode == "fast" 
    else accurate_process(data) if mode == "accurate" 
    else default_process(data)
)
```


## Условное выражение vs. Альтернативные подходы

### 1. Словари вместо условий
Иногда словари дают более читаемое решение:

```python
# Тернарный оператор для нескольких условий
result = (
    "A" if score >= 90 else
    "B" if score >= 80 else
    "C" if score >= 70 else
    "D" if score >= 60 else
    "F"
)

# Словарь с интервалами (более декларативно)
grade_map = {
    (90, 100): "A",
    (80, 89): "B",
    (70, 79): "C",
    (60, 69): "D",
    (0, 59): "F"
}
result = next(
    grade for (low, high), grade in grade_map.items() 
    if low <= score <= high
)
```

### 2. Лямбда-словари
Для сложных преобразований:

```python
# Тернарный оператор в лямбде
transform = lambda x: x*2 if x > 0 else x/2 if x < 0 else 0

# Лямбда-словарь
transform = {
    lambda x: x > 0: lambda x: x*2,
    lambda x: x < 0: lambda x: x/2,
    lambda x: True: lambda x: 0
}.get(lambda x: x, lambda x: 0)
```


## Практические рекомендации

### 1. Правило одной логической операции
```python
# Плохо: слишком много логики в одном тернарнике
result = x*2 + y if a > 0 and b < 10 or c == "test" else z/2 - w

# Хорошо: одна концептуальная операция
result = calculate_positive(x, y) if is_valid(x) else calculate_negative(z, w)
```

### 2. Используйте скобки для многострочной записи (или символ продолжения строки \)
```python
# Читаемая многострочная логика
classification = (
    "critical" if severity > 9 else
    "high" if severity > 7 else
    "medium" if severity > 4 else
    "low"
)

classification = \
    "critical" if severity > 9 else \
    "high" if severity > 7 else \
    "medium" if severity > 4 else \
    "low"
```

### 3. Комбинируйте с оператором walrus (:=)
```python
# Избегаем повторных вычислений
result = (
    process(value) if (value := calculate()) > threshold 
    else fallback(value)
)
```

### 4. Для сложных условий используйте функции-предикаты
```python
# Вместо сложного тернарника
is_eligible = lambda user: (
    user["age"] >= 18 and 
    user["status"] == "active" and
    user["score"] > 50
)

result = process(user) if is_eligible(user) else reject(user)
```
## Новые возможности языка (как `match/case`) добавляют функциональных черт:

```python
def evaluate(expr):
    match expr:
        case ["+", x, y]: return evaluate(x) + evaluate(y)
        case ["*", x, y]: return evaluate(x) * evaluate(y)
        case int(n): return n


## Почему тернарный оператор — ключ к чистому функциональному стилю

В функциональном программировании **всё является выражением**, возвращающим значение. Тернарный оператор `x if condition else y` — это единственный способ условной логики в Python, который остаётся выражением, а не оператором. Это критически важно для:

- Лямбда-выражений (где обычные `if` недоступны)
- Построения чистых функций без побочных эффектов
- Создания конвейеров преобразований без разрывов

```python
# Лямбда с тернарным оператором (работает)
safe_divide = lambda a, b: a/b if b != 0 else float('nan')

# Лямбда с обычным if (ошибка синтаксиса!)
# safe_divide = lambda a, b: 
#     if b != 0: 
#         return a/b 
#     else: 
#         return float('nan')
```
