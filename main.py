from tkinter import *
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
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



# Создаем главное окно
root = Tk()
root.geometry("1400x1000")
#root.resizable(False,False)



#меню ввода параметров
params_frame = ttk.Frame(height=800,width=600,borderwidth=1,relief=GROOVE)
params_frame.pack(anchor = NE,side = LEFT,padx=40,pady=40)

beet_batch_frame = ttk.Frame(height=150, width=180, master=params_frame)
beet_batch_frame.grid(row = 0, column = 0, pady = 10, padx = 10)
beet_batch_lab = ttk.Label(master=beet_batch_frame, anchor = NE, text ="Number of beet batches")
beet_batch_lab.pack()
beet_batch_ent = ttk.Entry(master = beet_batch_frame)
beet_batch_ent.pack()


num_expiriments_frame = ttk.Frame(height=150,width=180,master=params_frame)
num_expiriments_frame.grid(row=0,column = 1,padx =10 ,pady=10)
num_expiriments_lab = ttk.Label(master=num_expiriments_frame, anchor = NE, text ="Number of experiments")
num_expiriments_lab.pack()
num_expiriments_ent = ttk.Entry(master = num_expiriments_frame)
num_expiriments_ent.pack()


sugar_content_frame = ttk.Frame(height=100,width=180,master=params_frame)
sugar_content_frame.grid(columnspan = 2,sticky = EW,pady = 10,padx =10)
sugar_content_lab = ttk.Label(master=sugar_content_frame,text = "Sugar content before processing")
sugar_content_lab.pack()
parent_frame = ttk.Frame(master = sugar_content_frame,width=400,height=80)
parent_frame.pack(fill=BOTH,expand=True)
frame1 = ttk.Frame(master = parent_frame,width=100,height=40 )
frame1.pack(side = LEFT,padx = 50)
frame2 = ttk.Frame(master = parent_frame,width=100,height=40)
frame2.pack(side =RIGHT,padx = 50)

sugar_min_lab = ttk.Label(master=frame1,text= "min:")
sugar_max_lab = ttk.Label(master=frame2,text= "max:")
sugar_min_lab.pack()
sugar_max_lab.pack()
sugar_min_ent = ttk.Entry(master=frame1)
sugar_min_ent.pack()
sugar_max_ent = ttk.Entry(master=frame2)
sugar_max_ent.pack()



effect_of_inorganic_frame = ttk.Frame(height=100,width=180,master=params_frame)
effect_of_inorganic_frame.grid(columnspan = 2,sticky = EW,pady = 10,padx =10)
effect_of_inorganic_lab = ttk.Label(master=effect_of_inorganic_frame,text="consider the effects of inorganics")
# effect_of_inorganic_lab.pack()
effect_of_inorganic_chkbtn = ttk.Checkbutton(master=effect_of_inorganic_frame,text ="consider the effects of inorganics")
effect_of_inorganic_chkbtn.pack()


distribution_of_degradation = ttk.Frame(height=100,width=180,master=params_frame)
distribution_of_degradation.grid(columnspan = 2,sticky = EW,pady = 10,padx =10)
distribution_of_degradation_lab = ttk.Label(master= distribution_of_degradation,text = "Distribution of degradation")
distribution_of_degradation_lab.pack()

parent_frame1 = ttk.Frame(master = distribution_of_degradation,width=400,height=80)
parent_frame1.pack(fill=BOTH,expand=True)
frame12 = ttk.Frame(master = parent_frame1,width=100,height=40 )
frame12.pack(side = LEFT,padx = 50)
frame22 = ttk.Frame(master = parent_frame1,width=100,height=40)
frame22.pack(side =RIGHT,padx = 50)

sugar_min_lab = ttk.Label(master=frame12,text= "min:")
sugar_max_lab = ttk.Label(master=frame22,text= "max:")
sugar_min_lab.pack()
sugar_max_lab.pack()
sugar_min_ent = ttk.Entry(master=frame12)
sugar_min_ent.pack()
sugar_max_ent = ttk.Entry(master=frame22)
sugar_max_ent.pack()



#результаты
resulfts_frame = ttk.Frame(height=400,width=700,borderwidth=1,relief=SOLID,master=root)
resulfts_frame.pack(side =TOP,anchor = NE,padx=100,pady=40)

#график
plot_frame = ttk.Frame(height=150,width=150,borderwidth=1,relief=SOLID,master=root)
plot_frame.pack(side = BOTTOM,anchor = SE,padx=100,pady=40)

y = np.sin(np.linspace(0,2*np.pi,100))
fig,ax = plt.subplots()
ax.plot(np.linspace(0,1,100),y)

canvas = FigureCanvasTkAgg(fig,master=plot_frame)
canvas.draw()
canvas.get_tk_widget().pack(anchor = SE)

# Запускаем главный цикл
root.mainloop()
