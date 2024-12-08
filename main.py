from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import numpy as np
import scipy.optimize as so
import random


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


def calculate_results():
	global ax
	for i in range(0, N):
		greedy_results = greedy_algorithm(MATRIX, [0 for _ in range(0, MATRIX_SIZE)], 0, MATRIX_SIZE)
		ax.plot(np.linspace(0, 5, 5), greedy_results[1])


# def ChangeMatr(index):
# 	matterial = np.array([[22.15, 20.58, 21.08, 22.24, 22.52, 18.93], [7.05, 4.99, 4.88, 5.35, 7.05, 5.23],
# 						  [0.35, 0.38, 0.31, 0.21, 0.75, 0.82], [1.88, 1.91, 2.01, 1.58, 2.8, 2.72]])
#
# 	B = 0.12 * (matterial[1][index - 1] + matterial[2][index - 1]) + 0.24 * matterial[3][index - 1] + 0.48
#
# 	# global entry_matrix1
# 	'''matrix = []
# 	for i in range(int(entry_rows.get())):
# 		rows = []
# 		for j in range(int(entry_cols.get())):
# 			rows.append(float(entry_matrix[i][j].get()) - B)
# 		matrix.append(rows)
# 	#print(matrix)'''
#
# 	rows = int(entry_rows.get())
# 	cols = int(entry_cols.get())
#
# 	matrix = np.zeros((rows, cols))
# 	for i in range(rows):
# 		for j in range(cols):
# 			matrix[i][j] = float(entry_matrix[i][j].get()) - B
#
# 	return matrix
#
#
# def ChangeMatr1(index, matr):
# 	matterial = np.array([[22.15, 20.58, 21.08, 22.24, 22.52, 18.93], [7.05, 4.99, 4.88, 5.35, 7.05, 5.23],
# 						  [0.35, 0.38, 0.31, 0.21, 0.75, 0.82], [1.88, 1.91, 2.01, 1.58, 2.8, 2.72]])
#
# 	B = 0.12 * (matterial[1][index - 1] + matterial[2][index - 1]) + 0.24 * matterial[3][index - 1] + 0.48
#
# 	# global entry_matrix1
# 	'''matrix = []
# 	for i in range(int(entry_rows.get())):
# 		rows = []
# 		for j in range(int(entry_cols.get())):
# 			rows.append(float(entry_matrix[i][j].get()) - B)
# 		matrix.append(rows)
# 	#print(matrix)'''
#
# 	rows = int(entry_rows1.get())
# 	cols = int(entry_cols1.get())
#
# 	matrix = np.zeros((rows, cols))
# 	for i in range(rows):
# 		for j in range(cols):
# 			matrix[i][j] = matr[i][j] - B
#
# 	return matrix
#
#
# def calculate_result(matrix, algorithm, split):
# 	# � еализация выбранного алгоритма обработки матрицы
# 	# Здесь нужно вставить ваш код для обработки матрицы в соответствии с выбранным алгоритмом
# 	# В этом примере просто возвращается строковое представление матрицы
#
# 	batchCount = int(entry_rows.get())
# 	processingCount = int(entry_cols.get())
#
# 	col_val = []
#
# 	if (algorithm == "Венгерский алгоритм (max)"):
# 		result, col_val = HungarianAlgorithm(matrix, 1)
# 	elif (algorithm == "Венгерский алгоритм (min)"):
# 		result, col_val = HungarianAlgorithm(matrix)
# 	elif (algorithm == "Жадный алгоритм"):
# 		mark = [0] * batchCount
# 		result, col_val = GreedyAlgorithm(matrix, mark, 0, processingCount)
# 	elif (algorithm == "Бережливый алгоритм"):
# 		mark = [0] * batchCount
# 		result, col_val = LeanAlgorithm(matrix, mark, 0, processingCount)
# 	elif (algorithm == "Бережливо-жадный алгоритм" or algorithm == "Жадно-бережливый алгоритм"):
#
# 		if (algorithm == "Бережливо-жадный алгоритм"):
# 			result, col_val = LeanGreedyAlgorithm(matrix, batchCount, processingCount, int(split), 1)
# 		else:
# 			result, col_val = LeanGreedyAlgorithm(matrix, batchCount, processingCount, int(split))
# 	# elif (algorithm == "Влияние неорганики"):
# 	# entry_matrix1 = ChangeMatr(sel_sorts)
#
# 	# print(result)
# 	return result
#
#
# def on_calculate_clicked():
# 	# Получаем размеры матрицы
# 	rows = int(entry_rows.get())
# 	cols = int(entry_cols.get())
#
# 	global entry_matrix
# 	# print(entry.get(), entry_matrix[0,0].get())
#
# 	matrix = np.zeros((rows, cols))
# 	for i in range(rows):
# 		# row = []
# 		for j in range(cols):
# 			value = entry_matrix[i, j].get()
# 			# print("value =", value)
# 			matrix[i, j] = float(value)
# 	# print("matrix[i][j] =", matrix[i, j])
# 	# print(matrix[i, j], end = ', ')
# 	# row.append(value)
# 	# matrix.append(row)
# 	# print('\n', matrix)
#
# 	# print("splits:", *splits)
#
# 	# Получаем выбранный алгоритм из выпадающего списка
# 	selected_algorithm = algorithm_var.get()
# 	sel_split = splits_var.get()
#
# 	# print('sel_split =', sel_split)
#
# 	# Вызываем функцию для вычисления результата
# 	result = calculate_result(matrix, selected_algorithm, sel_split)
# 	# print(result)
#
# 	# Выводим результат в текстовое поле
# 	result_text.config(state="normal")
# 	result_text.delete(1.0, tk.END)  # Очищаем предыдущий результат
# 	result_text.insert(tk.END, result)
# 	result_text.config(state="disabled")
#

# def on_calculate_clicked1():
# 	# Получаем размеры матрицы
# 	rows = int(entry_rows.get())
# 	cols = int(entry_cols.get())
#
# 	global entry_matrix1
#
# 	matrix = np.zeros((rows, cols))
# 	for i in range(rows):
# 		# row = []
# 		for j in range(cols):
# 			value = entry_matrix1[i][j].get()
# 			matrix[i][j] = float(value)
# 	# row.append(value)
# 	# matrix.append(row)
# 	# print(matrix)
#
# 	# print("splits:", *splits)
#
# 	# Получаем выбранный алгоритм из выпадающего списка
# 	selected_algorithm = algorithm_var1.get()
# 	sel_split = splits_var1.get()
#
# 	# print('sel_split =', sel_split)
#
# 	# Вызываем функцию для вычисления результата
# 	result = calculate_result(matrix, selected_algorithm, sel_split)
#
# 	# Выводим результат в текстовое поле
# 	result_text1.config(state="normal")
# 	result_text1.delete(1.0, tk.END)  # Очищаем предыдущий результат
# 	result_text1.insert(tk.END, result)
# 	result_text1.config(state="disabled")

#
# def update_matrix():
# 	# ROW = int(entry_rows.get())
# 	COL = int(entry_cols.get())
#
# 	'''label_matrix = tk.Label(root, text="Matrix:")
# 	label_matrix.grid(row=1, column=0, padx=5, pady=5)
#
# 	# Создаем текстовые поля для ввода элементов матрицы
#
# 	global entry_matrix
# 	for i in range(ROW):  # Здесь использовано фиксированное количество строк для примера
# 		row_entries = []
# 		for j in range(COL):  # Здесь использовано фиксированное количество столбцов для примера
# 			entry = tk.Entry(root, width=5)
# 			entry.insert(tk.END, '0')
# 			entry.grid(row=i+1, column=j+1, padx=5, pady=5)
# 			row_entries.append(entry)
# 		entry_matrix.append(row_entries)'''
#
# 	# ЕСЛ�? ЭТО ЗАКОММЕНТ�?ТЬ �? УБ� АТЬ DISABLED В ОСНОВНОЙ П� ОГ� АММЕ П� �? ПЕ� ВОМ
# 	# ГЕНЕ� �?� ОВАН�?�? МАТ� �?ЦЫ, АЛГО� �?ТМЫ БУДУТ � АБОТАТЬ
#
# 	global entry_matrix
# 	# entry.destroy()
#
# 	rows = int(entry_rows.get())
# 	cols = int(entry_cols.get())
#
# 	for i in range(CONST_ROW):
# 		for j in range(CONST_COL):
# 			# entry = tk.Entry(root, width=5)
# 			if (i < rows and j < cols):
# 				entry_matrix[i, j].config(state="normal")
# 				entry_matrix[i, j].delete(0, tk.END)
# 				entry_matrix[i, j].insert(tk.END, '0')
# 			else:
# 				entry_matrix[i, j].delete(0, tk.END)
# 				entry_matrix[i, j].insert(tk.END, '0')
# 				entry_matrix[i, j].config(state="disable")
#
# 	global splits
# 	global splits_var
# 	# global splits_menu
# 	splits = [f"{i + 1}" for i in range(COL + 1)]  # int(entry_cols.get())
# 	splits_var = tk.StringVar(manualInputTab)
# 	splits_var.set(splits[0])  # Устанавливаем значение по умолчанию
# 	splits_menu = ttk.Combobox(manualInputTab, textvariable=splits_var, values=splits, width=2)
# 	splits_menu.grid(row=2, column=11, padx=5, pady=5)
#
# 	global splits_var1
# 	# global splits_menu
# 	splits = [f"{i + 1}" for i in range(COL + 1)]  # int(entry_cols.get())
# 	splits_var1 = tk.StringVar(manualInputTab)
# 	splits_var1.set(splits[0])  # Устанавливаем значение по умолчанию
# 	splits_menu1 = ttk.Combobox(manualInputTab, textvariable=splits_var1, values=splits, width=2)
# 	splits_menu1.grid(row=9, column=11, padx=5, pady=5)
#

# def second_matrix():
# 	index = int(sorts_var.get())
# 	result = ChangeMatr(index)
#
# 	global entry_matrix1
#
# 	rows = int(entry_rows.get())
# 	cols = int(entry_cols.get())
#
# 	for i in range(CONST_ROW):
# 		for j in range(CONST_COL):
# 			entry_matrix1[i, j].config(state="normal")
# 			if (i < rows and j < cols):
# 				entry_matrix1[i, j].delete(0, tk.END)
# 				entry_matrix1[i, j].insert(tk.END, result[i][j])
# 				entry_matrix1[i, j].config(state="readonly")
# 			else:
# 				entry_matrix1[i, j].delete(0, tk.END)
# 				entry_matrix1[i, j].insert(tk.END, '0')
# 				entry_matrix1[i, j].config(state="disable")
#
#
# def consider_inorganic():
# 	st = inorganic_enabled.get()
# 	if (st):
# 		label_sort.config(state='normal')
# 		sorts_menu1.config(state='normal')
# 	else:
# 		label_sort.config(state='disabled')
# 		sorts_menu1.config(state='disabled')
#

# def experiments():
# 	batchCount = int(entry_rows1.get())
# 	procCount = int(entry_cols1.get())
# 	sugarMin = float(entry_min.get())
# 	sugarMax = float(entry_max.get())
# 	degradeMin = float(entry_min1.get())
# 	degradeMax = float(entry_max1.get())
# 	split = int(procCount / 2)
#
# 	sort = 0
# 	state = str(sorts_menu1.cget('state'))
# 	if (state == "normal"):
# 		sort = int(sorts_var1.get())
# 		print("sort is")
# 	else:
# 		print("no sort")
#
# 	expCount = int(entry_exp.get())
#
# 	matrix = np.zeros((batchCount, procCount))
# 	for i in range(batchCount):
#
# 		for j in range(procCount):
# 			if (j == 0):
# 				matrix[i][j] = np.random.uniform(sugarMin, sugarMax)
# 			else:
# 				degrade = np.random.uniform(degradeMin, degradeMax)
# 				matrix[i][j] = matrix[i][j - 1] * degrade
#
# 	print("before:", *matrix)
#
# 	graphics(matrix, batchCount, procCount, expCount, split)
#
# 	# Влияние неорганики
# 	if (sort != 0):
# 		matrix = ChangeMatr1(sort, matrix)
# 		print("after", *matrix)
# 		graphics(matrix, batchCount, procCount, expCount, split)




# Константы
# Матрица для которой мы всю задачу и решаем, взята из методички - стр. 12
MATRIX = [[8, 4, 7, 10, 6],
		  [7, 5, 7, 5, 6],
		  [7, 6, 8, 8, 9],
		  [4, 2, 7, 6, 8],
		  [2, 4, 9, 9, 5]]
MATRIX_SIZE = 5  # Размер матрицы
N = 15  # Кол-во этапов (дней) эксперимента
NU = 7

fig, ax = plt.subplots()

#TODO добавить проверку на пустые поля и содержимое полей
def calculate_res():

	n = 5 #int(beet_batch_ent.get())   #10
	a_min = float(sugar_min_ent.get())  #22
	a_max = float(sugar_max_ent.get())  #44
	b_min = float(degradation_min_ent.get()) #0.85
	b_max = float(degradation_max_ent.get())   #1
	# print("b_min = ",b_min)
	# print("b_max = ", b_max)
	b_min_dozar = 1
	b_max_dozar = 1.15
	num_exp = int(num_experiments_ent.get())  #5

	max_hung = 0
	min_hung = 0
	greed = 0
	lean = 0
	lean_greedy = 0
	greedy_lean = 0

	for k in range(num_exp):
		# a - вектор изначального содержания сахара в каждой партии
		a = np.random.uniform(a_min, a_max, n)

		v = 0
		if dozarivanie_state.get():
			# print("ДОЗАРИВАНИЕ")
			v = np.random.randint(1, round(n / 2))
			# print('v=', v)

		# генерируем матрицу b - коэфф деградации
		b = np.zeros((n, n - 1))
		for i in range(0, v):
			b[:, i] = np.random.uniform(b_min_dozar, b_max_dozar, n)
		for i in range(v, n - 1):
			b[:, i] = np.random.uniform(b_min, b_max, n)

		# генерируем матрицу C, Cij - содеражние сахара в i партии на j этапе переработки
		c = np.zeros((n, n))
		c[:, 0] = a
		for i in range(0, n - 1):
			c[:, i + 1] = c[:, i] * b[:, i]

		K = (4.8, 7.05)
		Na = (0.21, 0.82)
		N = (1, .58, 2.8)
		I = (0.62, 0.64)

		inorganic_matrix = np.zeros((n, n))
		# длительность этапа - 7дней
		# ВЛИЯНИЕ НЕОРГАНИКИ

		if effect_of_inorganic.get():
			# print("ВЛИЯНИЕ НЕОРГАНИКИ")
			K = random.uniform(K[0], K[1])
			Na = random.uniform(Na[0], Na[1])
			N = random.uniform(N[0], N[1])
			I = random.uniform(I[0], I[1])
			I_mas = np.array([I * (1.029) ** (7 * i - 7) for i in range(0, n)])
			I_mas[0] = I
			for i in range(n):
				for j in range(n):
					# print(I_mas)
					inorganic_matrix[i][j] = 0.1541 * (K + Na) + 0.2159 * N + 0.9989 * I_mas[j] + 0.1967
		#через абсолютные значения
		#c = c - inorganic_matrix

		# print(c)
		# print(inorganic_matrix[0,:])



		# print(inorganic_matrix[0,:])
		# c=c - (c[:,j]*inorganic_matrix[0,:][j])/100

		#через относительные значения
		#скорее всего это правильно
		for i in range(n):
			for j in range(n):
				c[i][j] = c[i][j] - (c[i][j]*inorganic_matrix[0,:][j])/100

		# print(c)

		global ax, canvas, plt




		# вызываем алгоритмы
		max_hung += hungarian_algorithm(c, 1)[0]
		min_hung += hungarian_algorithm(c, 0)[0]
		greed += greedy_algorithm(c, [0 for _ in range(0, n)], 0, n)[0]
		lean += lean_algorithm(c, [0 for _ in range(0, n)], 0, n)[0]
		lean_greedy += lean_greedy_algorithm(c, n, n, round(n / 2), 1)[0]
		greedy_lean += lean_greedy_algorithm(c, n, n, round(n / 2), 0)[0]

	max_hung /= num_exp
	min_hung /= num_exp
	greed /= num_exp
	lean /= num_exp
	lean_greedy /= num_exp
	greedy_lean /= num_exp


	plt.cla()
	ax.bar(["max", "min", "жадный", "бережливый", "бережливо-жадный", "жадно-бережливый"],
		   [max_hung, min_hung, greed, lean, lean_greedy, greedy_lean])
	ax.set_title(f"средние показатели алгоритмов за {num_exp} экспериментов")
	plt.xticks(rotation=-10)
	plt.ylim(min_hung - min_hung * 0.03, max_hung + max_hung * 0.03)
	canvas.draw()
	result_label.config(text=f"Потери алгоритмов относительно максимума"
							 f"\n\nЖадный - {round((max_hung - greed) / max_hung * 100, 2)}%"
							 f"\nБережливый - {round((max_hung - lean) / max_hung * 100, 2)}%"
							 f"\nЖадно-бережливый - {round((max_hung - greedy_lean) / max_hung * 100, 2)}%"
							 f"\nБережливо-жадный - {round((max_hung - lean_greedy) / max_hung * 100, 2)}%")

	conclusions_label.config(text = f"\n\nВыводы\n\n Лучший алгоритм - {["жадный", "бережливый", "бережливо-жадный", "жадно-бережливый"][np.argmax([ greed, lean, lean_greedy, greedy_lean])]}")


# Создаем главное окно
root = Tk()
root.geometry("1600x900")


#
left = ttk.Frame(height=800, width=600,borderwidth=1, relief=GROOVE,master=root)
left.pack(side = LEFT,fill =Y)
right = ttk.Frame(height=800, width=600,borderwidth=1, relief=GROOVE,master=root)
right.pack(side = RIGHT,fill =Y)


# меню ввода параметров
params_frame = ttk.Frame(height=800, width=600, borderwidth=3, relief=GROOVE,master=left)
params_frame.pack(anchor=NE, padx=40, pady=40)

#
#root
#	left
#		params_frame
#			beet_batch_frame.grid()
#			num_experiments_frame.grid()
#			sugar_content_frame.grid()
#			effect_of_inorganic_frame.grid()
#				неорганика
#				дозаривание
#			distribution_of_degradation.grid()
#				деградиция
#
#	right
#
#внутренности менюшки -------------
beet_batch_frame = ttk.Frame(height=150, width=180, master=params_frame)
beet_batch_frame.grid(row=0, column=0, pady=10, padx=10)
beet_batch_lab = ttk.Label(master=beet_batch_frame, anchor=NE, text="Кол-во партий свёклы",font=("Arial", 14))
beet_batch_lab.pack()
beet_batch_ent = ttk.Entry(master=beet_batch_frame)
beet_batch_ent.pack()

num_experiments_frame = ttk.Frame(height=150, width=180, master=params_frame)
num_experiments_frame.grid(row=0, column=1, padx=10, pady=10)
num_experiments_lab = ttk.Label(master=num_experiments_frame, anchor=NE, text="Кол-во экспериментов",font=("Arial", 14))
num_experiments_lab.pack()
num_experiments_ent = ttk.Entry(master=num_experiments_frame)
num_experiments_ent.pack()

sugar_content_frame = ttk.Frame(height=100, width=180, master=params_frame)
sugar_content_frame.grid(columnspan=2, sticky=EW, pady=10, padx=10)
sugar_content_lab = ttk.Label(master=sugar_content_frame, text="Содержание сахара до обработки",font=("Arial", 14))
sugar_content_lab.pack()
parent_frame = ttk.Frame(master=sugar_content_frame, width=400, height=80)
parent_frame.pack(fill=BOTH, expand=True)
frame1 = ttk.Frame(master=parent_frame, width=100, height=40)
frame1.pack(side=LEFT, padx=50)
frame2 = ttk.Frame(master=parent_frame, width=100, height=40)
frame2.pack(side=RIGHT, padx=50)

sugar_min_lab = ttk.Label(master=frame1, text="min:",font=("Arial", 14))
sugar_max_lab = ttk.Label(master=frame2, text="max:",font=("Arial", 14))
sugar_min_lab.pack()
sugar_max_lab.pack()
sugar_min_ent = ttk.Entry(master=frame1)
sugar_min_ent.pack()
sugar_max_ent = ttk.Entry(master=frame2)
sugar_max_ent.pack()

distribution_of_degradation = ttk.Frame(height=100, width=180, master=params_frame)
distribution_of_degradation.grid(columnspan=2, sticky=EW, pady=10, padx=10)
distribution_of_degradation_lab = ttk.Label(master=distribution_of_degradation, text="Коэффициент деградации",font=("Arial", 14))
distribution_of_degradation_lab.pack()

effect_of_inorganic_frame = ttk.Frame(height=100, width=180, master=params_frame)
effect_of_inorganic_frame.grid(columnspan=2, sticky=EW, pady=10, padx=10)
# effect_of_inorganic_lab = ttk.Label(master=effect_of_inorganic_frame, text="Учитывать влияние неорганики")
# effect_of_inorganic_lab.pack()


effect_of_inorganic = IntVar()
def inorg_change():
	if effect_of_inorganic.get()==1:
		effect_of_inorganic.set(0)
	else:
		effect_of_inorganic.set(1)
effect_of_inorganic_chkbtn = Checkbutton(master=effect_of_inorganic_frame,
											 text="Учитывать влияние неорганики",variable=effect_of_inorganic,command=inorg_change,font=("Arial", 14))
effect_of_inorganic_chkbtn.pack()

dozarivanie_state = IntVar()
def dozar_change():
	if dozarivanie_state.get()==1:
		dozarivanie_state.set(0)
	else:
		dozarivanie_state.set(1)
dozarivanie = Checkbutton(master=effect_of_inorganic_frame,text="Учитывать дозаривание",variable=dozarivanie_state,command=dozar_change,font=("Arial", 14))
dozarivanie.pack()

# chbtn =

# distribution_of_degradation = ttk.Frame(height=100, width=180, master=params_frame)
# distribution_of_degradation.grid(columnspan=2, sticky=EW, pady=10, padx=10)
# distribution_of_degradation_lab = ttk.Label(master=distribution_of_degradation, text="Распределение деградации")
# distribution_of_degradation_lab.pack()


parent_frame1 = ttk.Frame(master=distribution_of_degradation, width=400, height=80)
parent_frame1.pack(fill=BOTH, expand=True)
frame12 = ttk.Frame(master=parent_frame1, width=100, height=40)
frame12.pack(side=LEFT, padx=50)
frame22 = ttk.Frame(master=parent_frame1, width=100, height=40)
frame22.pack(side=RIGHT, padx=50)

degradation_min_lab = ttk.Label(master=frame12, text="min:",font=("Arial",14))
degradation_max_lab = ttk.Label(master=frame22, text="max:",font=("Arial",14))
degradation_min_lab.pack()
degradation_max_lab.pack()
degradation_min_ent = ttk.Entry(master=frame12)
degradation_min_ent.pack()
degradation_max_ent = ttk.Entry(master=frame22)
degradation_max_ent.pack()
#-------------------



# результаты
results_frame = ttk.Frame(height=300, width=700, borderwidth=1, relief=SOLID, master=right)
results_frame.pack( padx=100, pady=40,fill = X)
result_label = ttk.Label(text="Потери алгоритмов относительно максимума"
							  "\n\nЖадный -"
							  "\nБережливый - "
							  "\nЖадно-бережливый - "
							  "\nБережливо-жадный - ", master=results_frame,font=("Arial", 14))
result_label.pack(anchor = NW)
# result_label.config(text="qwe")

# выводы
conclusions_frame= ttk.Frame(master=results_frame,height=150, width=300,borderwidth=2)
conclusions_frame.pack(anchor = NW)
conclusions_label = ttk.Label(text=f"\n\nВыводы\n\n Лучший алгоритм - ",master=conclusions_frame,font=("Arial", 14))
conclusions_label.pack(anchor = NW)

# график
plot_frame = ttk.Frame(height=150, width=150, borderwidth=1, relief=SOLID, master=right)
plot_frame.pack( padx=100, pady=40)

y = np.sin(np.linspace(0, 2 * np.pi, 100))
fig, ax = plt.subplots(figsize= (20,6),dpi =90)
#plt.xlim(0, 5)
#plt.ylim(0, 100)
# ax.plot(np.linspace(0, 1, 100), y)

canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.draw()
canvas.get_tk_widget().pack(anchor=SE)

# Кнопка расчёта результатов
# TODO исправить расположение кнопки :(   ->  (@^◡^)
button_frame = ttk.Frame(master=params_frame)
button_frame.grid(row=1, column=0)
btn = Button(master=left, text="Рассчитать", command=calculate_res,font=("Arial", 14))
btn.pack()

# Запускаем главный цикл
root.mainloop()
