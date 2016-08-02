
# coding: utf-8

# In[113]:

import re
import pandas as pd


# ## Работа с файлом TASK 1
# 
# На основе первого столбца данной матрицы необходимо проделать следующие шаги:
# * Если в строчках есть кавычки (два типа кавычек), то оставить только то, что находится внутри них
# * Все пробелы (в том числе, двойные, тройные и т.д.), знаки табуляции, переносы заменить на одиночный пробел
# * Удалить пробелы в начале, и в конце строки 
# * Удалить всё, что находится в скобках (вместе со скобками)
# * Удалить строчки, содержащие слова "жилой дом"или "таунхаус"
# * Удалить "ЖК"в началах строк
# * Сделать первую букву каждого слова в строках - заглавной
# * Извлечь уникальные значения и сделать таблицу датафреймом, экспортировать её в формате csv

# In[115]:

def replace_bad(x):
    x = x.replace('"', '')
    x = ' '.join(x.split())
    bad = ['-', '\t']
    for b in bad: x = x.replace(b, ' ')
    x = re.sub(r'\(.*\)', '', x)
    x = x[2:] if 'ЖК' in x[:2] else x    
    return x.strip().title()

df = pd.read_csv('task1.csv', sep=';')
df['estate_object_name'] = df.estate_object_name.apply(replace_bad)
df.drop_duplicates('estate_object_name').to_csv('task1_fixed.csv', index=False, sep=';', encoding='utf-8')


# ## Работа с файлом TASK 2
# 
# Каждый столбец матрицы разбить на большее количество столбцов по соответствующим разделителям
# * Из первого исходного столбца необходимо извлечь идентификатор, который идет в конце каждой записи
# * Также необходимо извлечь название городов и название метро для каждой строчки первого столбца
# * Сопоставить координаты из второго исходного столбца (располагаются в порядке X, Y ) - станциям метро и городам из первого исходного столбца используя уникальный идентификатор
# * Должен получиться следующий dataframe со столбцами: ("Город, Метро, ID, X, Y"), который необходимо экспортировать в csv-формате в кодировке UTF8
# * Подключить на Google disc следующий плагин: fusion tables. Импортировать туда полученный csv-файл
# * В столбце rows поменять формат переменной Y на location (two column location, latitude -Y , longitude - X) переключить вкладку на карту и выбрать в поле location - Y)
# * Далее карту нужно опубликовать, в Tools выбрать Publish, в Change visibility поменять Private на Anyone with the link, сохранить, и скопировать ссылку в всплывающем окне

# In[116]:

df = pd.read_csv('task2.csv', sep=';')
df.columns = ['text', 'coord']

def parse_text(x):    
    s = [i.strip().replace('"', '') for i in x.split(',')]
    if len(s) == 6:
        cnt, reg, city, line, station, ind = s
    else:
        reg = '-'
        try:
            cnt, city, line, station, ind = s            
        except:
            cnt, city, line, station, ind = ['-'] * 5
            print('Cannot parse metro:', s)
    line = line[:-6]
    station = station[6:]
    return pd.Series([cnt, reg, city, line, station, ind])

def parse_coord(x):
    sp = x.split('|')
    return pd.Series([sp[0], sp[1], sp[2]])

df[['cnt', 'reg', 'city', 'line', 'station', 'item_id']] = df.text.apply(parse_text)
df[['x', 'y', 'val']] = df.coord.apply(parse_coord)
df_final = df[(df.city != '-') & (df.city != 'Северный административный округ')][['item_id', 'city', 'station', 'x', 'y']].copy()
df_final.columns = ['ID', 'Город', 'Метро', 'X', 'Y']
df_final.to_csv('task2_fixed.csv', index=False, encoding='utf-8')

print('\nLink to share map: https://www.google.com/fusiontables/DataSource?docid=1kcEqMroQUBgQv0im9C8LdiXxEvTURsHUn_ymkePH')


# ### Задача 1
# 
# Что больше: $e^\pi$ или $\pi^e$?
# 
# #### Решение
# 
# Возьмем логарифм от обоих выражений и немного преобразуем: $$\ln(e^\pi) \quad v \quad \ln(\pi^e)$$
# $$\pi \quad v \quad e\ln(\pi)$$
# $$\frac{\pi}{e} \quad v \quad \ln(\pi) - \ln(e) + \ln(e)$$
# $$\frac{\pi}{e} \quad v \quad \ln\left(\frac{\pi}{e}\right) + 1$$
# $$\frac{\pi}{e}-1 \quad v \quad \ln\left(\frac{\pi}{e}\right)$$
# 
# Пусть $f(x)=x-1, \quad g(x)=\ln(x)$. Тогда нам надо сравнить $f(x)$ и $g(x)$ при $x=\frac{\pi}{e}$. Сделаем замену переменных $x=t+1$. Тогда $f(t) = t, \quad g(t)=\ln(1+t)$. Но $g(t) = \ln(1+t) = t - \frac{t^2}{2} + \frac{t^3}{3} + \ldots < t$. Значит, $f(x) > g(x)$ и $e^\pi > \pi^e$.
