import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Работа с библиотекой numpy

a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a.shape)  #вывод колличества строк и столбцов

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)  # вывод таблицы из списков
print(a.shape)   #вывод колличества строк и столбцов

print(np.linspace(0, 20, num=5))  #вывод линейного массива через указанный интервал

#Работа с библиотекой pandas

s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)
print()

data = {
    'Name': ['Olga', 'Dmitry', 'Igor'],
    'Age': [28, 17, 42],
    'City': ['Moscow', 'Piter', 'Rostov'],
    'Gender': ['woman', 'man', 'man']
}

df = pd.DataFrame(data)

# Вывод первых двух строк DataFrame
print(df.head(2))
print()

# Добавление нового столбца
df['Salary'] = [70000, 80000, 60000]

# Фильтрация данных: выбор людей старше 25 лет
filtered_df = df[df['Age'] > 25]

print(filtered_df)

#Работа с библиотекой Matplotlib

fig, ax = plt.subplots()             # Создаём оси координат.
ax.plot([1, 2, 3, 4], [2, 4, 2, 3])  # Наносим реперные точки.
plt.show()                           # Выводим рисунок.


def my_plotter(ax, data1, data2, param_dict): # Вспомогательная функция для построения графика.
    out = ax.plot(data1, data2, **param_dict)
    return out

data1, data2, data3, data4 = np.random.randn(4, 100)  # создаем 4 случайных набора данных
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.7))
my_plotter(ax1, data1, data2, {'marker': 'x'})
my_plotter(ax2, data3, data4, {'marker': 'o'})
plt.show()


data1 = 1
data2 = 3
fig, ax = plt.subplots(figsize=(5, 2.7))
ax.scatter(data1, data2, s=50, facecolor='y', edgecolor='r')  #маркер жёлтый, а край маркера красного цвета
plt.show()