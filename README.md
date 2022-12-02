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

## Задание 1
- Исходный проект
[![](https://github.com/MikhaillVodolaginn/DA-in-GameDev-Lab5/blob/main/Снимок%20экрана%202022-12-02%20в%2022.16.01.png?raw=true)
- Установка пакетов в Pakage Manager
[![](https://github.com/MikhaillVodolaginn/DA-in-GameDev-Lab5/blob/main/Снимок%20экрана%202022-12-02%20в%2022.16.11.png?raw=true)
- Установка необходимого окружения
[![](https://github.com/MikhaillVodolaginn/DA-in-GameDev-Lab5/blob/main/Снимок%20экрана%202022-12-02%20в%2022.19.10.png?raw=true)
- Запуск сцены с 12 префабами


https://user-images.githubusercontent.com/103418305/205361765-32000f3a-2e30-4662-b7a7-bfa4c254d9be.mov


- Результаты обучения сохранены
[![](https://github.com/MikhaillVodolaginn/DA-in-GameDev-Lab5/blob/main/Снимок%20экрана%202022-12-02%20в%2022.28.47.png?raw=true)
- Установка tensorflow и запуск tensorboard с помощью командной строки
[![](https://github.com/MikhaillVodolaginn/DA-in-GameDev-Lab5/blob/main/Снимок%20экрана%202022-12-02%20в%2022.49.16.png?raw=true)
- tensorboard
[![](https://github.com/MikhaillVodolaginn/DA-in-GameDev-Lab5/blob/main/Снимок%20экрана%202022-12-02%20в%2023.06.33.png?raw=true)

## Задание 2
### Подробно опишите каждую строку файла конфигурации нейронной сети, доступного в папке с файлами проекта по ссылке.

```
behaviors: # Основной раздел файла конфигурации, набор конфигураций для каждого поведения в сцене.
    RollerBall: # Название конфигурации.
    trainer_type: ppo # (Proximal Policy Optimization), тип используемого тренера или режим обучения.
hyperparameters: # Раздел настройки гиперпараметров для агента.
    batch_size: 10 # Количество опытов на каждой итерации.
    buffer_size: 100 # Количество опыта, которое нужно набрать перед обновлением модели. Обычно большее buffer_size значение соответствует более стабильным обновлениям обучения.
    learning_rate: 3.0e-4 # Начальная скорость обучения.
    beta: 5.0e-4 # Сила регуляризации энтропии(мера хаоса), которая делает политику «более случайной».
    epsilon: 0.2 # Насколько быстро политика может развиваться во время обучения.
    lambd: 0.99 #  Насколько агент полагается на свою текущую оценку стоимости при расчете обновленной оценки стоимости. Низкие значения соответствуют большему полаганию на текущую оценку ценности, а высокие значения соответствуют большему полаганию на фактические вознаграждения.
    num_epoch: 3 # Количество проходов через буфер опыта, при выполнении оптимизации.
    learning_rate_schedule: linear # определяет как скорость обучения изменяется с течением времени, linear линейно уменьшает скорость
network_settings: # Раздел, содержащий настройки нейронной сети.
    normalize: false # Нормализация к входным данным векторного наблюдения.
    hidden_units: 128 # Количество нейронов в скрытых слоях сети.
    num_layers: 2 # Количество скрытых слоёв в нейронной сети.
reward_signals: # Раздел позволяет задавать настройки для внешних и внутренних сигналов вознаграждения. Каждый сигнал вознаграждения должен определять как минимум два параметра: strength и gamma.
    extrinsic: # Внешние сигналы вознаграждения(если они есть).
    gamma: 0.99 # Коэффициент дисконтирования для будущих вознаграждений. Как далеко в будущем агент должен заботиться о возможных вознаграждениях, должен быть строго меньше 1.
    strength: 1.0 # Коэффициент на который умножается вознаграждение (по умолчанию = 1.0).
max_steps: 500000 # Максимальное количество проходов.
time_horizon: 64 # Временные рамки.
summary_freq: 10000 # Количество опыта, который необходимо собрать перед созданием и отображением статистики.
```

## Задание 3
### Доработайте сцену и обучите ML-Agent таким образом, чтобы шар перемещался между двумя кубами разного цвета. Кубы должны, как и в первом задании, случайно изменять координаты на плоскости.

- Измененный C# скрипт
```
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public GameObject Target1;
    public GameObject Target2;
    private bool target1Collected;
    private bool target2Collected;

    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }
        Target1.transform.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
        Target2.transform.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
        Target1.SetActive(true);
        Target2.SetActive(true);
        target1Collected = false;
        target2Collected = false;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target1.transform.localPosition);
        sensor.AddObservation(Target2.transform.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(target1Collected);
        sensor.AddObservation(target2Collected);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }

    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);
        float distanceToTarget1 = Vector3.Distance(this.transform.localPosition, Target1.transform.localPosition);
        float distanceToTarget2 = Vector3.Distance(this.transform.localPosition, Target2.transform.localPosition);
        if (!target1Collected & distanceToTarget1 < 1.42f)
        {
            target1Collected = true;
            Target1.SetActive(false);
        }
        if (!target2Collected & distanceToTarget2 < 1.42f)
        {
            target2Collected = true;
            Target2.SetActive(false);
        }
        if (target1Collected & target2Collected)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
```

- Обучение нового скрипта, среднее вознаграждение 0,989
[![](https://github.com/MikhaillVodolaginn/DA-in-GameDev-Lab3/blob/main/Снимок%20экрана%202022-11-01%20в%2016.35.42.png?raw=true)

## Выводы

В данной лабораторной работе я научился подключать mlagent и pytorch, а также впервые работал в консоли с помощью анаконды.

Игровой баланс - одно из важнейших понятий в играх, так как от него зависит равенство взаимодействий тех или иных объектов, правил, механик.

Хороший баланс должен поддерживать не слишком простой уровень сложности, чтобы не было слишком скучно и не слишком высокий, чтобы не возникало отторжения. Игрок должен ощущать, что игра честна по отношению к нему. В играх существует огромное разнообразие объектов и добиться синергии между ними очень сложно, особенно в асимметричных играх, где у каждого игрока качественно разные способности и стили игры, необходимо учесть много факторов, однако нейросети могут с этим помочь, например, могут подстраивать степень сложности окружающего мира под игрока, поэтому развивая себя игрок одновременно развивает и мир вокруг. Нейросети могут анализировать пикрейт персонажей в многопользовательской игре и уменьшать характеристики тех героев, на которых он повышен

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
