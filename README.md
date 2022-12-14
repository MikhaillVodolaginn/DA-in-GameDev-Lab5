# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #5 выполнил:
- Водолагин Михаил Алексеевич
- РИ210942
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Интеграция экономической системы в проект Unity и обучение ML-Agent

## Задание 1, 2
### Измените параметры файла.yaml-агента.
- Исходный проект
[![](https://github.com/MikhaillVodolaginn/DA-in-GameDev-Lab5/blob/main/Снимок%20экрана%202022-12-02%20в%2022.16.01.png?raw=true)
- Установка пакетов в Pakage Manager
[![](https://github.com/MikhaillVodolaginn/DA-in-GameDev-Lab5/blob/main/Снимок%20экрана%202022-12-02%20в%2022.16.11.png?raw=true)
- Установка необходимого окружения
[![](https://github.com/MikhaillVodolaginn/DA-in-GameDev-Lab5/blob/main/Снимок%20экрана%202022-12-02%20в%2022.19.10.png?raw=true)
- Запуск сцены с 12 префабами


https://user-images.githubusercontent.com/103418305/205361765-32000f3a-2e30-4662-b7a7-bfa4c254d9be.mov


- Установка tensorflow и запуск tensorboard с помощью командной строки
[![](https://github.com/MikhaillVodolaginn/DA-in-GameDev-Lab5/blob/main/Снимок%20экрана%202022-12-02%20в%2022.49.16.png?raw=true)
- tensorboard
[![](https://github.com/MikhaillVodolaginn/DA-in-GameDev-Lab5/blob/main/Снимок%20экрана%202022-12-02%20в%2023.06.33.png?raw=true)

## Задание 3
### Определить какие параметры и как влияют на обучение модели. Опишите результаты, выведенные в TensorBoard.
- изменим переменную lambd c 0.95 на 0.75:
[![](https://github.com/MikhaillVodolaginn/DA-in-GameDev-Lab5/blob/main/Снимок%20экрана%202022-12-02%20в%2023.08.53.png?raw=true)
График Extrinsic Reward стал более пологим, то есть значенuе внешнего вознаграждения изменяется в меньших диапазонах, остальные без видимых изменений
- изменим параметр num_layers c 2 до 5
[![](https://github.com/MikhaillVodolaginn/DA-in-GameDev-Lab5/blob/main/Снимок%20экрана%202022-12-02%20в%2023.04.41.png?raw=true)
Не произошло никаких видимых изменений
- изменим переменную epsilon с 0.2 на 0.3
[![](https://github.com/MikhaillVodolaginn/DA-in-GameDev-Lab5/blob/main/Снимок%20экрана%202022-12-02%20в%2023.08.06.png?raw=true)
График Extrinsic Reward стал монотонно расти, то есть внешнее вознаграждение пропорционально epsilon, график Entropy стал почти параллелен горизонтальной оси, то есть энтропия с учетом небольшой погрешности можно считать константой, остальные графики превратились в точку

По итогу уменьшение lambd стабилизирует внешнее вознаграждение, то есть делает обучение модели более плавным, num_layers не оказывает никакого влияния, а увеличение epsilon увеличивает внешнее вознаграждение и скорость обучения соответственно


## Выводы

В данной лабораторной работе я научился подключать tensorflow, а также впервые работал с экономичской системой с помощью Unity и MLAgent.

Я узнал как может изменяться инфляция в игре в зависимости от параметров. Tensorboard очень удобен для отслеживания динамики инляции при различных конфигурациях.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

Vodolagin Mikhail
