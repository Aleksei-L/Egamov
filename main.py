import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so


# Венгерский алгоритм
def hungarian_algorithm(matrix, mode: int):
	"""
	Венгерский алгоритм
	:param matrix: матрица к которой необходимо применить алгоритм
	:param mode: задаёт что мы должны найти: минимум(0) или максимум(1)
	:return: Кортеж, где 0-й элемент - max/min целевой функции (summa), а остальные n элементов - решение матрицы
	"""
	# В матрицу строк и столбцов записываются соответствующие индексы,
	# из которых мы получим элементы, удовлетворяющие нашему условию (min/max)
	if mode == 0:
		row_index, col_index = so.linear_sum_assignment(matrix, maximize=False)
	else:
		row_index, col_index = so.linear_sum_assignment(matrix, maximize=True)

	summa = matrix[row_index, col_index].sum()  # Максимум/минимум целевой функции

	row_index = list(row_index)
	col_index = list(col_index)
	col_ind_2 = []
	row_ind_2 = []

	while len(col_index) != 0:
		min_elem = col_index[0]
		index = 0
		for i in range(len(col_index)):
			if min_elem > col_index[i]:
				min_elem = col_index[i]
				index = i
		col_ind_2.append(col_index.pop(index))
		row_ind_2.append(row_index.pop(index))

	col_val = [matrix[row_ind_2[i], col_ind_2[i]] for i in range(len(col_ind_2))]
	return summa, col_val


# Жадный алгоритм
def greedy_algorithm(matrix, arr: list, start_column: int, end_column: int):
	"""
	Жадный алгоритм: в каждом столбце выбирает наибольший элемент
	:param matrix: матрица к которой необходимо применить алгоритм
	:param arr: массив из 0/1 в котором указывается была ли обработана строка или нет
	:param start_column: номер столбца матрицы matrix с которого мы начнём применение алгоритма
	:param end_column: номер столбца матрицы matrix на котором мы закончим применение алгоритма
	:return: кортеж, где 0-й элемент - max/min целевой функции, а остальные n элементов - решение матрицы
	"""
	result = 0
	result_arr = [0] * (end_column - start_column)
	k = 0

	for j in range(start_column, end_column):
		max_elem = None
		for i in range(len(arr)):
			if arr[i] == 0:
				if (max_elem is None) or (matrix[i][j] > max_elem):
					max_elem = matrix[i][j]
					k = i

		arr[k] = 1
		result_arr[j] = max_elem
		result += max_elem
	return result, result_arr


# Бережливый алгоритм
def lean_algorithm(matrix, arr: list, start_column: int, end_column: int):
	"""
	Бережливый алгоритм: в каждом столбце выбирает наименьший элемент
	:param matrix: матрица к которой необходимо применить алгоритм
	:param arr: массив из 0/1 в котором указывается была ли обработана строка или нет
	:param start_column: номер столбца матрицы matrix с которого мы начнём применение алгоритма
	:param end_column: номер столбца матрицы matrix на котором мы закончим применение алгоритма
	:return: кортеж, где 0-й элемент - max/min целевой функции, а остальные n элементов - решение матрицы
	"""
	result = 0
	result_arr = [0] * (end_column - start_column)
	k = 0

	for j in range(start_column, end_column):
		min_elem = None
		for i in range(len(arr)):
			if arr[i] == 0:
				if (min_elem is None) or (matrix[i][j] < min_elem):
					min_elem = matrix[i][j]
					k = i
		arr[k] = 1
		result_arr[j] = min_elem
		result += min_elem
	return result, result_arr


# Бережливо-жадный или жадно-бережливый алгоритм
def lean_greedy_algorithm(matrix, batch_count: int, processing_count: int, split: int, mode: int):
	"""
	Бережливо-жадный алгоритм: применяет бережливый алгоритм к первым split-1 столбцам и жадный алгоритм к оставшимся столбцам
	Жадно-бережливый алгоритм: применяет жадный алгоритм к первым split-1 столбцам и бережливый алгоритм к оставшимся столбцам
	:param matrix: матрица к которой необходимо применить алгоритм
	:param batch_count: кол-во строк матрицы matrix
	:param processing_count: кол-во столбцов матрицы matrix
	:param split: номер столбца с которого начнёт действовать жадный алгоритм
	:param mode: выбор алгоритма: Ж-Б при mode=0 и Б-Ж при mode=1
	:return: кортеж, где 0-й элемент - max/min целевой функции, а остальные n элементов - решение матрицы
	"""
	matrix_1 = np.zeros((batch_count, processing_count))
	matrix_2 = np.zeros((batch_count, processing_count))

	b1 = split - 1
	b2 = processing_count - b1

	for i in range(batch_count):
		for j1 in range(b1):
			matrix_1[i][j1] = matrix[i][j1]
		for j2 in range(b1, processing_count):
			matrix_2[i][j2 - split + 1] = matrix[i][j2]

	mark = [0] * batch_count

	if mode == 1:
		summa_1, first_arr = lean_algorithm(matrix_1, mark, 0, b1)
		summa_2, second_arr = greedy_algorithm(matrix_2, mark, 0, b2)
	else:
		summa_1, first_arr = greedy_algorithm(matrix_1, mark, 0, b1)
		summa_2, second_arr = lean_algorithm(matrix_2, mark, 0, b2)

	summa = summa_1 + summa_2
	col_arr = first_arr + second_arr

	return summa, col_arr


def change_matrix(index: int):
	"""
	TODO
	Эта функция рассчитывает число, которое запишется во все ячейки второй матрицы
	:param index: номер сорта
	:return:
	"""
	material = np.array([
		[22.15, 20.58, 21.08, 22.24, 22.52, 18.93],
		[7.05, 4.99, 4.88, 5.35, 7.05, 5.23],
		[0.35, 0.38, 0.31, 0.21, 0.75, 0.82],
		[1.88, 1.91, 2.01, 1.58, 2.8, 2.72]
	])

	b = 0.12 * (material[1][index - 1] + material[2][index - 1]) + 0.24 * material[3][index - 1] + 0.48

	rows = int(entry_rows.get())
	cols = int(entry_cols.get())

	matrix = np.zeros((rows, cols))
	for i in range(rows):
		for j in range(cols):
			matrix[i][j] = float(entry_matrix[i][j].get()) - b

	return matrix


def change_matrix_1(index, matr):
	"""
	TODO
	Какой-то аналог функции change_matrix
	:param index:
	:param matr:
	:return:
	"""
	material = np.array([
		[22.15, 20.58, 21.08, 22.24, 22.52, 18.93],
		[7.05, 4.99, 4.88, 5.35, 7.05, 5.23],
		[0.35, 0.38, 0.31, 0.21, 0.75, 0.82],
		[1.88, 1.91, 2.01, 1.58, 2.8, 2.72]
	])

	b = 0.12 * (material[1][index - 1] + material[2][index - 1]) + 0.24 * material[3][index - 1] + 0.48

	rows = int(entry_rows1.get())
	cols = int(entry_cols1.get())

	matrix = np.zeros((rows, cols))
	for i in range(rows):
		for j in range(cols):
			matrix[i][j] = matr[i][j] - b

	return matrix


def calculate_result(matrix, algorithm, split):
	# Реализация выбранного алгоритма обработки матрицы
	# Здесь нужно вставить ваш код для обработки матрицы в соответствии с выбранным алгоритмом
	# В этом примере просто возвращается строковое представление матрицы

	batch_count = int(entry_rows.get())
	processing_count = int(entry_cols.get())

	if algorithm == "Венгерский алгоритм (max)":
		result, col_val = hungarian_algorithm(matrix, mode=1)
	elif algorithm == "Венгерский алгоритм (min)":
		result, col_val = hungarian_algorithm(matrix, mode=0)
	elif algorithm == "Жадный алгоритм":
		mark = [0] * batch_count
		result, col_val = greedy_algorithm(matrix, mark, 0, processing_count)
	elif algorithm == "Бережливый алгоритм":
		mark = [0] * batch_count
		result, col_val = lean_algorithm(matrix, mark, 0, processing_count)
	elif algorithm == "Бережливо-жадный алгоритм":
		result, col_val = lean_greedy_algorithm(matrix, batch_count, processing_count, int(split), mode=1)
	elif algorithm == "Жадно-бережливый алгоритм":
		result, col_val = lean_greedy_algorithm(matrix, batch_count, processing_count, int(split), mode=0)

	return result


def on_calculate_clicked():
	# Получаем размеры матрицы
	rows = int(entry_rows.get())
	cols = int(entry_cols.get())

	global entry_matrix
	# print(entry.get(), entry_matrix[0,0].get())

	matrix = np.zeros((rows, cols))
	for i in range(rows):
		# row = []
		for j in range(cols):
			value = entry_matrix[i, j].get()
			# print("value =", value)
			matrix[i, j] = float(value)
	# print("matrix[i][j] =", matrix[i, j])
	# print(matrix[i, j], end = ', ')
	# row.append(value)
	# matrix.append(row)
	# print('\n', matrix)

	# print("splits:", *splits)

	# Получаем выбранный алгоритм из выпадающего списка
	selected_algorithm = algorithm_var.get()
	sel_split = splits_var.get()

	# print('sel_split =', sel_split)

	# Вызываем функцию для вычисления результата
	result = calculate_result(matrix, selected_algorithm, sel_split)
	# print(result)

	# Выводим результат в текстовое поле
	result_text.config(state="normal")
	result_text.delete(1.0, tk.END)  # Очищаем предыдущий результат
	result_text.insert(tk.END, result)
	result_text.config(state="disabled")


def on_calculate_clicked_1():
	# Получаем размеры матрицы
	rows = int(entry_rows.get())
	cols = int(entry_cols.get())

	global entry_matrix_1

	matrix = np.zeros((rows, cols))
	for i in range(rows):
		# row = []
		for j in range(cols):
			value = entry_matrix_1[i][j].get()
			matrix[i][j] = float(value)
	# row.append(value)
	# matrix.append(row)
	# print(matrix)

	# print("splits:", *splits)

	# Получаем выбранный алгоритм из выпадающего списка
	selected_algorithm = algorithm_var1.get()
	sel_split = splits_var1.get()

	# print('sel_split =', sel_split)

	# Вызываем функцию для вычисления результата
	result = calculate_result(matrix, selected_algorithm, sel_split)

	# Выводим результат в текстовое поле
	result_text1.config(state="normal")
	result_text1.delete(1.0, tk.END)  # Очищаем предыдущий результат
	result_text1.insert(tk.END, result)
	result_text1.config(state="disabled")


def update_matrix():
	# ROW = int(entry_rows.get())
	col = int(entry_cols.get())

	'''label_matrix = tk.Label(root, text="Matrix:")
	label_matrix.grid(row=1, column=0, padx=5, pady=5)

	# Создаем текстовые поля для ввода элементов матрицы

	global entry_matrix
	for i in range(ROW):  # Здесь использовано фиксированное количество строк для примера
		row_entries = []
		for j in range(col):  # Здесь использовано фиксированное количество столбцов для примера
			entry = tk.Entry(root, width=5)
			entry.insert(tk.END, '0')
			entry.grid(row=i+1, column=j+1, padx=5, pady=5)
			row_entries.append(entry)
		entry_matrix.append(row_entries)'''

	# ЕСЛИ ЭТО ЗАКОММЕНТИТЬ И УБРАТЬ DISABLED В ОСНОВНОЙ ПРОГРАММЕ ПРИ ПЕРВОМ
	# ГЕНЕРИРОВАНИИ МАТРИЦЫ, АЛГОРИТМЫ БУДУТ РАБОТАТЬ

	global entry_matrix
	# entry.destroy()

	rows = int(entry_rows.get())
	cols = int(entry_cols.get())

	for i in range(CONST_ROW):
		for j in range(CONST_COL):
			# entry = tk.Entry(root, width=5)
			if i < rows and j < cols:
				entry_matrix[i, j].config(state="normal")
				entry_matrix[i, j].delete(0, tk.END)
				entry_matrix[i, j].insert(tk.END, '0')
			else:
				entry_matrix[i, j].delete(0, tk.END)
				entry_matrix[i, j].insert(tk.END, '0')
				entry_matrix[i, j].config(state="disable")

	global splits
	global splits_var
	# global splits_menu
	splits = [f"{i + 1}" for i in range(col + 1)]  # int(entry_cols.get())
	splits_var = tk.StringVar(manual_input_tab)
	splits_var.set(splits[0])  # Устанавливаем значение по умолчанию
	splits_menu = ttk.Combobox(manual_input_tab, textvariable=splits_var, values=splits, width=2)
	splits_menu.grid(row=2, column=11, padx=5, pady=5)

	global splits_var1
	# global splits_menu
	splits = [f"{i + 1}" for i in range(col + 1)]  # int(entry_cols.get())
	splits_var1 = tk.StringVar(manual_input_tab)
	splits_var1.set(splits[0])  # Устанавливаем значение по умолчанию
	splits_menu1 = ttk.Combobox(manual_input_tab, textvariable=splits_var1, values=splits, width=2)
	splits_menu1.grid(row=9, column=11, padx=5, pady=5)


def second_matrix():
	index = int(sorts_var.get())
	result = change_matrix(index)

	global entry_matrix_1

	rows = int(entry_rows.get())
	cols = int(entry_cols.get())

	for i in range(CONST_ROW):
		for j in range(CONST_COL):
			entry_matrix_1[i, j].config(state="normal")
			if i < rows and j < cols:
				entry_matrix_1[i, j].delete(0, tk.END)
				entry_matrix_1[i, j].insert(tk.END, result[i][j])
				entry_matrix_1[i, j].config(state="readonly")
			else:
				entry_matrix_1[i, j].delete(0, tk.END)
				entry_matrix_1[i, j].insert(tk.END, '0')
				entry_matrix_1[i, j].config(state="disable")


def consider_inorganic():
	st = inorganic_enabled.get()
	if st:
		label_sort.config(state='normal')
		sorts_menu1.config(state='normal')
	else:
		label_sort.config(state='disabled')
		sorts_menu1.config(state='disabled')


def experiments():
	batch_count = int(entry_rows1.get())
	proc_count = int(entry_cols1.get())
	sugar_min = float(entry_min.get())
	sugar_max = float(entry_max.get())
	degrade_min = float(entry_min1.get())
	degrade_max = float(entry_max1.get())
	split = int(proc_count / 2)

	sort = 0
	state = str(sorts_menu1.cget('state'))
	if state == "normal":
		sort = int(sorts_var1.get())
		print("sort is")
	else:
		print("no sort")

	exp_count = int(entry_exp.get())

	matrix = np.zeros((batch_count, proc_count))
	for i in range(batch_count):
		for j in range(proc_count):
			if j == 0:
				matrix[i][j] = np.random.uniform(sugar_min, sugar_max)
			else:
				degrade = np.random.uniform(degrade_min, degrade_max)
				matrix[i][j] = matrix[i][j - 1] * degrade

	print("before:", *matrix)

	graphics(matrix, batch_count, proc_count, exp_count, split)

	# Влияние неорганики
	if sort != 0:
		matrix = change_matrix_1(sort, matrix)
		print("after", *matrix)
		graphics(matrix, batch_count, proc_count, exp_count, split)


def graphics(matrix, batch_count, proc_count, exp_count, split):
	algorithm_get = np.zeros((len(algorithms), proc_count))

	for i in range(exp_count):

		summa, cols_val = hungarian_algorithm(matrix, mode=1)
		for j in range(batch_count):
			algorithm_get[0][j] += cols_val[j]

		summa, cols_val = hungarian_algorithm(matrix, mode=0)
		for j in range(batch_count):
			algorithm_get[1][j] += cols_val[j]

		mark = [0] * batch_count
		summa, cols_val = greedy_algorithm(matrix, mark, 0, proc_count)
		for j in range(batch_count):
			algorithm_get[2][j] += cols_val[j]

		mark = [0] * batch_count
		summa, cols_val = lean_algorithm(matrix, mark, 0, proc_count)
		for j in range(batch_count):
			algorithm_get[3][j] += cols_val[j]

		summa, cols_val = lean_greedy_algorithm(matrix, batch_count, proc_count, split, mode=1)
		for j in range(batch_count):
			algorithm_get[4][j] += cols_val[j]

		summa, cols_val = lean_greedy_algorithm(matrix, batch_count, proc_count, split, mode=0)
		for j in range(batch_count):
			algorithm_get[5][j] += cols_val[j]

	for i in range(len(algorithms)):
		for j in range(batch_count):
			algorithm_get[i][j] /= exp_count
	proc = [i for i in range(proc_count)]

	get_sum = np.cumsum(algorithm_get[0])
	plt.plot(proc, get_sum, color="black", label="Венгерский макс", linestyle='--')

	get_sum = np.cumsum(algorithm_get[1])
	plt.plot(proc, get_sum, color="green", label="Венгерский мин", linestyle='-.')

	get_sum = np.cumsum(algorithm_get[2])
	plt.plot(proc, get_sum, color="orange", label="Жадный алгоритм", linestyle='dotted')

	get_sum = np.cumsum(algorithm_get[3])
	plt.plot(proc, get_sum, color="blue", label="Бережливый алгоритм", linestyle='-')

	get_sum = np.cumsum(algorithm_get[4])
	plt.plot(proc, get_sum, color="purple", label="Береж/Жадн алгоритм", linestyle=':')

	get_sum = np.cumsum(algorithm_get[5])
	plt.plot(proc, get_sum, color="red", label="Жадн/Береж алгоритм", linestyle='--')

	plt.xticks(proc)
	plt.legend()
	plt.title("Средние значения алг. на каждом этапе переработки")
	plt.xlabel("Номер переработки, N")
	plt.ylabel("Средние значения алгоритмов, S")
	plt.show()


CONST_ROW = 5
CONST_COL = 5

# Создаем главное окно
color_txt = "#ffffff"
color = "#caacfa"
color_btn = "#e0e0e0"
color_lbl = "#af7dff"
root = tk.Tk()

# Создаем ТабКонтрол:
tab_control = ttk.Notebook(root)

# Затем создаем две вкладки
manual_input_tab = tk.Frame(tab_control, bg=color)
automatic_input_tab = tk.Frame(tab_control, bg=color)

# Затем добавляем эти две вкладки в наш ТабКонтрол
tab_control.add(manual_input_tab, text="Ручной ввод")
tab_control.add(automatic_input_tab, text="Автоматический ввод")

tab_control.grid(row=0, column=0)

root.geometry('865x410')
root.resizable(0, 0)
root["bg"] = color
root.title("Решение прикладных задач дискретной оптимизации")

# Создаем и размещаем виджеты в РУЧНОМ ВВОДЕ
label_ent = tk.Label(manual_input_tab, text="Задайте размер матрицы:", bg=color_lbl)
label_ent.grid(row=1, column=0, padx=0, pady=5, columnspan=2)

'''label_rows = tk.Label(manualInputTab, text="Строки:", bg=colorLbl)
label_rows.grid(row=2, column=0, padx=0, pady=5, sticky='w')
entry_rows = tk.Entry(manualInputTab, width = 10)
entry_rows.insert(tk.END, '0')
entry_rows.grid(row=2, column=1, padx=5, pady=5, sticky='w')

label_cols = tk.Label(manualInputTab, text="Столбцы:", bg=colorLbl)
label_cols.grid(row=3, column=0, padx=0, pady=5, sticky='w')
entry_cols = tk.Entry(manualInputTab, width = 10)
entry_cols.insert(tk.END, '0')
entry_cols.grid(row=3, column=1, padx=5, pady=5, sticky='w')'''

entry_size = tk.Entry(manual_input_tab, width=10)
entry_size.insert(tk.END, '0')
entry_size.grid(row=2, column=0, padx=5, pady=5, columnspan=2)

entry_cols = entry_size
entry_rows = entry_size

submit_button = tk.Button(manual_input_tab, text="Подтвердить", command=update_matrix, bg=color_btn)
submit_button.grid(row=3, column=0, padx=10, pady=5, columnspan=2)

# label_matrix = tk.Label(root, text="Matrix:")
# label_matrix.grid(row=1, column=0, padx=5, pady=5)

label_2 = tk.Label(manual_input_tab, text=" ", bg=color)
label_2.grid(row=1, column=2, padx=30, pady=5)

label_matrix = tk.Label(manual_input_tab, text="Исходная матрица:", bg=color_lbl)
label_matrix.grid(row=1, column=3, pady=5, columnspan=5)

# Создаем текстовые поля для ввода элементов матрицы
entry_matrix = np.zeros((CONST_ROW, CONST_COL), dtype=object)
for i in range(CONST_ROW):  # Здесь использовано фиксированное количество строк для примера
	row_entries = []
	for j in range(CONST_COL):  # Здесь использовано фиксированное количество столбцов для примера
		entry = tk.Entry(manual_input_tab, width=5)
		entry.config(state="normal")
		entry.insert(tk.END, '0')
		entry.config(state="disabled")
		entry.grid(row=i + 2, column=j + 3, padx=5, pady=5)
		entry_matrix[i, j] = entry
# row_entries.append(entry)
# entry_matrix.append(row_entries)

label_matrix_1 = tk.Label(manual_input_tab, text="Влияние неорганики:", bg=color_lbl)
label_matrix_1.grid(row=8, column=3, padx=5, pady=5, columnspan=5)

# КОПИЯ МАТРИЦЫ
entry_matrix_1 = np.zeros((CONST_ROW, CONST_COL), dtype=object)
for i in range(CONST_ROW):  # Здесь использовано фиксированное количество строк для примера
	row_entries_1 = []
	for j in range(CONST_COL):  # Здесь использовано фиксированное количество столбцов для примера
		entry_1 = tk.Entry(manual_input_tab, width=5)
		entry_1.config(state="normal")
		entry_1.insert(tk.END, '0')
		entry_1.config(state="disabled")
		entry_1.grid(row=i + 9, column=j + 3, padx=5, pady=5)
		entry_matrix_1[i, j] = entry_1

label_2 = tk.Label(manual_input_tab, text="", bg=color)
label_2.grid(row=1, column=9, padx=30, pady=5)

label_alg = tk.Label(manual_input_tab, text="Выбор алгоритма:", bg=color_lbl)
label_alg.grid(row=1, column=10, padx=5, pady=5)

label_split = tk.Label(manual_input_tab, text="Период:", bg=color_lbl)
label_split.grid(row=1, column=11, padx=5, pady=5)

# Создаем выпадающий список с алгоритмами
algorithms = ["Венгерский алгоритм (max)", "Венгерский алгоритм (min)", "Жадный алгоритм", "Бережливый алгоритм",
			  "Бережливо-жадный алгоритм", "Жадно-бережливый алгоритм"]
algorithm_var = tk.StringVar(manual_input_tab)
algorithm_var.set(algorithms[0])  # Устанавливаем значение по умолчанию
algorithm_menu = ttk.Combobox(manual_input_tab, textvariable=algorithm_var, values=algorithms, width=30)
algorithm_menu.grid(row=2, column=10, padx=5, pady=5, sticky='w')

# Создаем выпадающий список с периодами
splits = [f"{i + 1}" for i in range(int(entry_cols.get()) + 1)]  # int(entry_cols.get())
# splits = ["1", "2", "3", "4", "5"]
splits_var = tk.StringVar(manual_input_tab)
splits_var.set(splits[0])  # Устанавливаем значение по умолчанию
splits_menu = ttk.Combobox(manual_input_tab, textvariable=splits_var, values=splits, width=2)
splits_menu.grid(row=2, column=11, padx=5, pady=5)

# Кнопка для вычисления результата
calculate_button = tk.Button(manual_input_tab, text="Рассчитать", command=on_calculate_clicked, bg=color_btn)
calculate_button.grid(row=2, column=12, padx=5, pady=5, columnspan=2)

# Создаем текстовое поле для вывода результата
result_text = tk.Text(manual_input_tab, height=1, width=25, state="disabled")
result_text.grid(row=3, column=10, padx=5, pady=5, sticky='w')

label_sort = tk.Label(manual_input_tab, text="Выбор № сорта:", bg=color_lbl)
label_sort.grid(row=8, column=0, padx=5, pady=5, columnspan=2)

# Создаем выпадающий список с сортами
sorts = [f"{i + 1}" for i in range(6)]  # int(entry_cols.get())
# splits = ["1", "2", "3", "4", "5"]
sorts_var = tk.StringVar(manual_input_tab)
sorts_var.set(sorts[0])  # Устанавливаем значение по умолчанию
sorts_menu = ttk.Combobox(manual_input_tab, textvariable=sorts_var, values=sorts, width=2)
sorts_menu.grid(row=9, column=0, padx=1, pady=5, columnspan=2)

# Кнопка для вычисления результата
calculate_button = tk.Button(manual_input_tab, text="Применить", command=second_matrix, bg=color_btn)
calculate_button.grid(row=10, column=0, padx=1, pady=5, columnspan=2)

label_alg1 = tk.Label(manual_input_tab, text="Выбор алгоритма:", bg=color_lbl)
label_alg1.grid(row=8, column=10, padx=5, pady=5)

label_split = tk.Label(manual_input_tab, text="Период:", bg=color_lbl)
label_split.grid(row=8, column=11, padx=5, pady=5)

# Создаем выпадающий список с алгоритмами
algorithm_var1 = tk.StringVar(manual_input_tab)
algorithm_var1.set(algorithms[0])  # Устанавливаем значение по умолчанию
algorithm_menu1 = ttk.Combobox(manual_input_tab, textvariable=algorithm_var1, values=algorithms, width=30)
algorithm_menu1.grid(row=9, column=10, padx=5, pady=5, sticky='w')

# Создаем выпадающий список с периодами
splits_var1 = tk.StringVar(manual_input_tab)
splits_var1.set(splits[0])  # Устанавливаем значение по умолчанию
splits_menu1 = ttk.Combobox(manual_input_tab, textvariable=splits_var1, values=splits, width=2)
splits_menu1.grid(row=9, column=11, padx=5, pady=5)

# Кнопка для вычисления результата
calculate_button1 = tk.Button(manual_input_tab, text="Рассчитать", command=on_calculate_clicked_1, bg=color_btn)
calculate_button1.grid(row=9, column=12, padx=5, pady=5)

# Создаем текстовое поле для вывода результата
result_text1 = tk.Text(manual_input_tab, height=1, width=25, state="disabled")
result_text1.grid(row=10, column=10, padx=5, pady=5, sticky='w')

# Создаем и размещаем виджеты в АВТОМАТИЧЕСКОМ ВВОДЕ
label_rows1 = tk.Label(automatic_input_tab, text=" ", bg=color)
label_rows1.grid(row=1, column=0, padx=200, pady=0, sticky='e')

label_rows1 = tk.Label(automatic_input_tab, text="Кол-во партий и переработок:", bg=color_lbl)
label_rows1.grid(row=2, column=0, padx=0, pady=9, sticky='e')
entry_rows1 = tk.Entry(automatic_input_tab, width=10)
entry_rows1.insert(tk.END, '0')
entry_rows1.grid(row=2, column=1, padx=5, pady=9, sticky='w')

entry_cols1 = entry_rows1

label_range = tk.Label(automatic_input_tab, text="Диапазон сахаристости партий до переработок", bg=color_lbl)
label_range.grid(row=3, column=0, padx=0, pady=9, sticky='e')

label_min = tk.Label(automatic_input_tab, text="min:", bg=color_lbl)
label_min.grid(row=4, column=0, padx=0, pady=9, sticky='e')
entry_min = tk.Entry(automatic_input_tab, width=10)
entry_min.insert(tk.END, '0')
entry_min.grid(row=4, column=1, padx=5, pady=9, sticky='w')

label_max = tk.Label(automatic_input_tab, text="max:", bg=color_lbl)
label_max.grid(row=4, column=2, padx=0, pady=9, sticky='e')
entry_max = tk.Entry(automatic_input_tab, width=10)
entry_max.insert(tk.END, '0')
entry_max.grid(row=4, column=3, padx=5, pady=9, sticky='w')

label_range1 = tk.Label(automatic_input_tab, text="Диапазон деградации:", bg=color_lbl)
label_range1.grid(row=5, column=0, padx=0, pady=9, sticky='e')

label_min1 = tk.Label(automatic_input_tab, text="min:", bg=color_lbl)
label_min1.grid(row=6, column=0, padx=0, pady=9, sticky='e')
entry_min1 = tk.Entry(automatic_input_tab, width=10)
entry_min1.insert(tk.END, '0')
entry_min1.grid(row=6, column=1, padx=5, pady=9, sticky='w')

label_max1 = tk.Label(automatic_input_tab, text="max:", bg=color_lbl)
label_max1.grid(row=6, column=2, padx=0, pady=9, sticky='e')
entry_max1 = tk.Entry(automatic_input_tab, width=10)
entry_max1.insert(tk.END, '0')
entry_max1.grid(row=6, column=3, padx=5, pady=9, sticky='w')

# label_space = tk.Label(automaticInputTab, text=" ", bg=color)
# label_space.grid(row=8, column=0, padx=0, pady=5, sticky='e')

label_inorganic = tk.Label(automatic_input_tab, text="Учитывать при выходе сахара влияние неорганики", bg=color_lbl)
label_inorganic.grid(row=7, column=0, padx=0, pady=9, sticky='e')

inorganic_enabled = tk.IntVar()
checkbox_inorganic = tk.Checkbutton(automatic_input_tab, variable=inorganic_enabled, command=consider_inorganic, bg=color)
checkbox_inorganic.grid(row=7, column=1, padx=5, pady=9, sticky="w")

label_sort = tk.Label(automatic_input_tab, text="Выбор сорта:", bg=color_lbl, state='disabled')
label_sort.grid(row=8, column=0, padx=0, pady=9, sticky='e')

# Создаем выпадающий список с сортами
# sorts1 = [f"{i + 1}" for i in range(6)] # int(entry_cols.get())
# splits = ["1", "2", "3", "4", "5"]
sorts_var1 = tk.StringVar(automatic_input_tab)
sorts_var1.set(sorts[0])  # Устанавливаем значение по умолчанию
sorts_menu1 = ttk.Combobox(automatic_input_tab, textvariable=sorts_var1, values=sorts, width=2, state='disabled')
sorts_menu1.grid(row=8, column=1, padx=5, pady=9, sticky='w')

label_exp = tk.Label(automatic_input_tab, text="Кол-во экспериментов:", bg=color_lbl)
label_exp.grid(row=9, column=0, padx=0, pady=9, sticky='e')
entry_exp = tk.Entry(automatic_input_tab, width=10)
entry_exp.insert(tk.END, '0')
entry_exp.grid(row=9, column=1, padx=5, pady=9, sticky='w')

# Кнопка для вычисления результата
calculate_button1 = tk.Button(automatic_input_tab, text="Рассчитать", command=experiments, bg=color_btn)
calculate_button1.grid(row=10, column=0, padx=65, pady=9, columnspan=3, sticky='e')

# Запускаем главный цикл
root.mainloop()
