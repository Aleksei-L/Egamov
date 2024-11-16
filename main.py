import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so


# Венгерский алгоритм
def HungarianAlgorithm(matrix, mode=0):
	# в матрицу строк и столбцов записываются соответствующие индексы,
	# из которых мы получим элементы, удовлетворяющие нашему условию (min/max)
	if (mode == 0):
		rowInd, colInd = so.linear_sum_assignment(matrix)
	# print(rowInd, "&&", colInd)
	# print("min")
	else:
		rowInd, colInd = so.linear_sum_assignment(matrix, True)
	# print(rowInd, "&&", colInd)
	# print("max")

	summa = matrix[rowInd, colInd].sum()  # максимум/минимум целевой функции
	# print(summa)

	rowInd = list(rowInd)
	colInd = list(colInd)
	colInd2 = []
	rowInd2 = []

	while (len(colInd) != 0):
		minElem = colInd[0]
		index = 0
		for i in range(len(colInd)):
			if (minElem > colInd[i]):
				minElem = colInd[i]
				index = i
		colInd2.append(colInd.pop(index))
		rowInd2.append(rowInd.pop(index))

	col_val = [matrix[rowInd2[i], colInd2[i]] for i in range(len(colInd2))]
	return summa, col_val


# Жадный алгоритм
def GreedyAlgorithm(matr, arr, a, b):
	res = 0
	resArr = [0] * (b - a)
	k = 0

	for j in range(a, b):
		# print("new col")
		'''k = 0
		while (True):
			if (k == len(arr)): 
				return res
			if (arr[k] == 0):
				maxElem = matr[k][j]
				# print('here')
				break
			k += 1
			# print(j, k)
			# time.sleep(5)'''
		maxElem = None
		for i in range(len(arr)):
			# print("i)", arr[i], "== 0?")
			if (arr[i] == 0):
				# if ((matr[i][j] > maxElem) and (arr[i] == 0)):
				if (maxElem == None or (matr[i][j] > maxElem)):
					maxElem = matr[i][j]
					k = i

		arr[k] = 1
		resArr[j] = maxElem
		res += maxElem
	# print(*res)
	return res, resArr


# Бережливый алгоритм
def LeanAlgorithm(matr, arr, a, b):
	res = 0
	resArr = [0] * (b - a)
	k = 0

	for j in range(a, b):
		'''k = 0
		while (True):
			if (k == len(arr)): 
				return res
			if (arr[k] == 0):
				minElem = matr[k][j]
				break
			k += 1'''

		minElem = None
		# print('size =', len(arr))
		for i in range(len(arr)):
			if (arr[i] == 0):
				# if ((matr[i][j] > maxElem) and (arr[i] == 0)):
				if (minElem == None or (matr[i][j] < minElem)):
					minElem = matr[i][j]
					k = i

		'''for i in range(len(arr)):
			if ((matr[i][j] < minElem) and (arr[i] == 0)):
				minElem = matr[i][j]
				k = i'''

		arr[k] = 1
		resArr[j] = minElem
		res += minElem
	return res, resArr


def LeanGreedyAlgorithm(matrix, batchCount, processingCount, split, mode=0):
	summa = 0
	summa1 = 0
	summa2 = 0

	matr1 = np.zeros((batchCount, processingCount))
	matr2 = np.zeros((batchCount, processingCount))

	b1 = split - 1
	b2 = processingCount - b1

	for i in range(batchCount):
		for j1 in range(b1):
			matr1[i][j1] = matrix[i][j1]
		for j2 in range(b1, processingCount):
			matr2[i][j2 - split + 1] = matrix[i][j2]

	mark = [0] * batchCount

	firstArr = [0] * b1
	secondArr = [0] * b2

	# print("mode =", mode)
	if (mode == 1):
		summa1, firstArr = LeanAlgorithm(matr1, mark, 0, b1)
		# print("lean =", summa1)
		# print(*mark)
		summa2, secondArr = GreedyAlgorithm(matr2, mark, 0, b2)
	# print("greedy =", summa2)
	# print(*mark)
	else:
		summa1, firstArr = GreedyAlgorithm(matr1, mark, 0, b1)
		# print("lean =", summa1)
		# print(*mark)
		summa2, secondArr = LeanAlgorithm(matr2, mark, 0, b2)
	# print("greedy =", summa2)
	# print(*mark)

	summa = summa1 + summa2
	colArr = firstArr + secondArr

	return summa, colArr


def ChangeMatr(index):
	matterial = np.array([[22.15, 20.58, 21.08, 22.24, 22.52, 18.93], [7.05, 4.99, 4.88, 5.35, 7.05, 5.23],
						  [0.35, 0.38, 0.31, 0.21, 0.75, 0.82], [1.88, 1.91, 2.01, 1.58, 2.8, 2.72]])

	B = 0.12 * (matterial[1][index - 1] + matterial[2][index - 1]) + 0.24 * matterial[3][index - 1] + 0.48

	# global entry_matrix1
	'''matrix = []
	for i in range(int(entry_rows.get())):
		rows = []
		for j in range(int(entry_cols.get())):
			rows.append(float(entry_matrix[i][j].get()) - B)
		matrix.append(rows)
	#print(matrix)'''

	rows = int(entry_rows.get())
	cols = int(entry_cols.get())

	matrix = np.zeros((rows, cols))
	for i in range(rows):
		for j in range(cols):
			matrix[i][j] = float(entry_matrix[i][j].get()) - B

	return matrix


def ChangeMatr1(index, matr):
	matterial = np.array([[22.15, 20.58, 21.08, 22.24, 22.52, 18.93], [7.05, 4.99, 4.88, 5.35, 7.05, 5.23],
						  [0.35, 0.38, 0.31, 0.21, 0.75, 0.82], [1.88, 1.91, 2.01, 1.58, 2.8, 2.72]])

	B = 0.12 * (matterial[1][index - 1] + matterial[2][index - 1]) + 0.24 * matterial[3][index - 1] + 0.48

	# global entry_matrix1
	'''matrix = []
	for i in range(int(entry_rows.get())):
		rows = []
		for j in range(int(entry_cols.get())):
			rows.append(float(entry_matrix[i][j].get()) - B)
		matrix.append(rows)
	#print(matrix)'''

	rows = int(entry_rows1.get())
	cols = int(entry_cols1.get())

	matrix = np.zeros((rows, cols))
	for i in range(rows):
		for j in range(cols):
			matrix[i][j] = matr[i][j] - B

	return matrix


def calculate_result(matrix, algorithm, split):
	# � еализация выбранного алгоритма обработки матрицы
	# Здесь нужно вставить ваш код для обработки матрицы в соответствии с выбранным алгоритмом
	# В этом примере просто возвращается строковое представление матрицы

	batchCount = int(entry_rows.get())
	processingCount = int(entry_cols.get())

	col_val = []

	if (algorithm == "Венгерский алгоритм (max)"):
		result, col_val = HungarianAlgorithm(matrix, 1)
	elif (algorithm == "Венгерский алгоритм (min)"):
		result, col_val = HungarianAlgorithm(matrix)
	elif (algorithm == "Жадный алгоритм"):
		mark = [0] * batchCount
		result, col_val = GreedyAlgorithm(matrix, mark, 0, processingCount)
	elif (algorithm == "Бережливый алгоритм"):
		mark = [0] * batchCount
		result, col_val = LeanAlgorithm(matrix, mark, 0, processingCount)
	elif (algorithm == "Бережливо-жадный алгоритм" or algorithm == "Жадно-бережливый алгоритм"):

		if (algorithm == "Бережливо-жадный алгоритм"):
			result, col_val = LeanGreedyAlgorithm(matrix, batchCount, processingCount, int(split), 1)
		else:
			result, col_val = LeanGreedyAlgorithm(matrix, batchCount, processingCount, int(split))
	# elif (algorithm == "Влияние неорганики"):
	# entry_matrix1 = ChangeMatr(sel_sorts)

	# print(result)
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


def on_calculate_clicked1():
	# Получаем размеры матрицы
	rows = int(entry_rows.get())
	cols = int(entry_cols.get())

	global entry_matrix1

	matrix = np.zeros((rows, cols))
	for i in range(rows):
		# row = []
		for j in range(cols):
			value = entry_matrix1[i][j].get()
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
	COL = int(entry_cols.get())

	'''label_matrix = tk.Label(root, text="Matrix:")
	label_matrix.grid(row=1, column=0, padx=5, pady=5)

	# Создаем текстовые поля для ввода элементов матрицы

	global entry_matrix
	for i in range(ROW):  # Здесь использовано фиксированное количество строк для примера
		row_entries = []
		for j in range(COL):  # Здесь использовано фиксированное количество столбцов для примера
			entry = tk.Entry(root, width=5)
			entry.insert(tk.END, '0')
			entry.grid(row=i+1, column=j+1, padx=5, pady=5)
			row_entries.append(entry)
		entry_matrix.append(row_entries)'''

	# ЕСЛ�? ЭТО ЗАКОММЕНТ�?ТЬ �? УБ� АТЬ DISABLED В ОСНОВНОЙ П� ОГ� АММЕ П� �? ПЕ� ВОМ
	# ГЕНЕ� �?� ОВАН�?�? МАТ� �?ЦЫ, АЛГО� �?ТМЫ БУДУТ � АБОТАТЬ

	global entry_matrix
	# entry.destroy()

	rows = int(entry_rows.get())
	cols = int(entry_cols.get())

	for i in range(CONST_ROW):
		for j in range(CONST_COL):
			# entry = tk.Entry(root, width=5)
			if (i < rows and j < cols):
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
	splits = [f"{i + 1}" for i in range(COL + 1)]  # int(entry_cols.get())
	splits_var = tk.StringVar(manualInputTab)
	splits_var.set(splits[0])  # Устанавливаем значение по умолчанию
	splits_menu = ttk.Combobox(manualInputTab, textvariable=splits_var, values=splits, width=2)
	splits_menu.grid(row=2, column=11, padx=5, pady=5)

	global splits_var1
	# global splits_menu
	splits = [f"{i + 1}" for i in range(COL + 1)]  # int(entry_cols.get())
	splits_var1 = tk.StringVar(manualInputTab)
	splits_var1.set(splits[0])  # Устанавливаем значение по умолчанию
	splits_menu1 = ttk.Combobox(manualInputTab, textvariable=splits_var1, values=splits, width=2)
	splits_menu1.grid(row=9, column=11, padx=5, pady=5)


def second_matrix():
	index = int(sorts_var.get())
	result = ChangeMatr(index)

	global entry_matrix1

	rows = int(entry_rows.get())
	cols = int(entry_cols.get())

	for i in range(CONST_ROW):
		for j in range(CONST_COL):
			entry_matrix1[i, j].config(state="normal")
			if (i < rows and j < cols):
				entry_matrix1[i, j].delete(0, tk.END)
				entry_matrix1[i, j].insert(tk.END, result[i][j])
				entry_matrix1[i, j].config(state="readonly")
			else:
				entry_matrix1[i, j].delete(0, tk.END)
				entry_matrix1[i, j].insert(tk.END, '0')
				entry_matrix1[i, j].config(state="disable")


def consider_inorganic():
	st = inorganic_enabled.get()
	if (st):
		label_sort.config(state='normal')
		sorts_menu1.config(state='normal')
	else:
		label_sort.config(state='disabled')
		sorts_menu1.config(state='disabled')


def experiments():
	batchCount = int(entry_rows1.get())
	procCount = int(entry_cols1.get())
	sugarMin = float(entry_min.get())
	sugarMax = float(entry_max.get())
	degradeMin = float(entry_min1.get())
	degradeMax = float(entry_max1.get())
	split = int(procCount / 2)

	sort = 0
	state = str(sorts_menu1.cget('state'))
	if (state == "normal"):
		sort = int(sorts_var1.get())
		print("sort is")
	else:
		print("no sort")

	expCount = int(entry_exp.get())

	matrix = np.zeros((batchCount, procCount))
	for i in range(batchCount):

		for j in range(procCount):
			if (j == 0):
				matrix[i][j] = np.random.uniform(sugarMin, sugarMax)
			else:
				degrade = np.random.uniform(degradeMin, degradeMax)
				matrix[i][j] = matrix[i][j - 1] * degrade

	print("before:", *matrix)

	graphics(matrix, batchCount, procCount, expCount, split)

	# Влияние неорганики
	if (sort != 0):
		matrix = ChangeMatr1(sort, matrix)
		print("after", *matrix)
		graphics(matrix, batchCount, procCount, expCount, split)


def graphics(matrix, batchCount, procCount, expCount, split):
	algorithmGet = np.zeros((len(algorithms), procCount))

	for i in range(expCount):

		summa, cols_val = HungarianAlgorithm(matrix, 1)
		for j in range(batchCount):
			algorithmGet[0][j] += cols_val[j]

		summa, cols_val = HungarianAlgorithm(matrix)
		for j in range(batchCount):
			algorithmGet[1][j] += cols_val[j]

		mark = [0] * batchCount
		summa, cols_val = GreedyAlgorithm(matrix, mark, 0, procCount)
		for j in range(batchCount):
			algorithmGet[2][j] += cols_val[j]

		mark = [0] * batchCount
		summa, cols_val = LeanAlgorithm(matrix, mark, 0, procCount)
		for j in range(batchCount):
			algorithmGet[3][j] += cols_val[j]

		summa, cols_val = LeanGreedyAlgorithm(matrix, batchCount, procCount, split, 1)
		for j in range(batchCount):
			algorithmGet[4][j] += cols_val[j]

		summa, cols_val = LeanGreedyAlgorithm(matrix, batchCount, procCount, split)
		for j in range(batchCount):
			algorithmGet[5][j] += cols_val[j]

	for i in range(len(algorithms)):
		for j in range(batchCount):
			algorithmGet[i][j] /= expCount
	proc = [i for i in range(procCount)]

	getSum = np.cumsum(algorithmGet[0])
	plt.plot(proc, getSum, color="black", label="Венгерский макс", linestyle='--')

	getSum = np.cumsum(algorithmGet[1])
	plt.plot(proc, getSum, color="green", label="Венгерский мин", linestyle='-.')

	getSum = np.cumsum(algorithmGet[2])
	plt.plot(proc, getSum, color="orange", label="Жадный алгоритм", linestyle='dotted')

	getSum = np.cumsum(algorithmGet[3])
	plt.plot(proc, getSum, color="blue", label="Бережливый алгоритм", linestyle='-')

	getSum = np.cumsum(algorithmGet[4])
	plt.plot(proc, getSum, color="purple", label="Береж/Жадн алгоритм", linestyle=':')

	getSum = np.cumsum(algorithmGet[5])
	plt.plot(proc, getSum, color="red", label="Жадн/Береж алгоритм", linestyle='--')

	plt.xticks(proc)
	plt.legend()
	plt.title("Средние значения алг. на каждом этапе переработки")
	plt.xlabel("Номер переработки, N")
	plt.ylabel("Средние значения алгоритмов, S")
	plt.show()


CONST_ROW = 5
CONST_COL = 5

# Создаем главное окно
colorTxt = "#ffffff"
color = "#caacfa"
colorBtn = "#e0e0e0"
colorLbl = "#af7dff"
root = tk.Tk()

# Создаем ТабКонтрол:
tabControl = ttk.Notebook(root)

# Затем создаем две вкладки
manualInputTab = tk.Frame(tabControl, bg=color)
automaticInputTab = tk.Frame(tabControl, bg=color)

# Затем добавляем эти две вкладки в наш ТабКонтрол
tabControl.add(manualInputTab, text="� учной ввод")
tabControl.add(automaticInputTab, text="Автоматический ввод")

tabControl.grid(row=0, column=0)

root.geometry('865x410')
root.resizable(0, 0)
root["bg"] = color
root.title("� ешение прикладных задач дискретной оптимизации")

# Создаем и размещаем виджеты в � УЧНОМ ВВОДЕ
label_ent = tk.Label(manualInputTab, text="Задайте размер матрицы:", bg=colorLbl)
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

entry_size = tk.Entry(manualInputTab, width=10)
entry_size.insert(tk.END, '0')
entry_size.grid(row=2, column=0, padx=5, pady=5, columnspan=2)

entry_cols = entry_size
entry_rows = entry_size

submit_button = tk.Button(manualInputTab, text="Подтвердить", command=update_matrix, bg=colorBtn)
submit_button.grid(row=3, column=0, padx=10, pady=5, columnspan=2)

# label_matrix = tk.Label(root, text="Matrix:")
# label_matrix.grid(row=1, column=0, padx=5, pady=5)

label_2 = tk.Label(manualInputTab, text=" ", bg=color)
label_2.grid(row=1, column=2, padx=30, pady=5)

label_matrix = tk.Label(manualInputTab, text="�?сходная матрица:", bg=colorLbl)
label_matrix.grid(row=1, column=3, pady=5, columnspan=5)

# Создаем текстовые поля для ввода элементов матрицы
entry_matrix = np.zeros((CONST_ROW, CONST_COL), dtype=object)
for i in range(CONST_ROW):  # Здесь использовано фиксированное количество строк для примера
	row_entries = []
	for j in range(CONST_COL):  # Здесь использовано фиксированное количество столбцов для примера
		entry = tk.Entry(manualInputTab, width=5)
		entry.config(state="normal")
		entry.insert(tk.END, '0')
		entry.config(state="disabled")
		entry.grid(row=i + 2, column=j + 3, padx=5, pady=5)
		entry_matrix[i, j] = entry
# row_entries.append(entry)
# entry_matrix.append(row_entries)

label_matrix1 = tk.Label(manualInputTab, text="Влияние неорганики:", bg=colorLbl)
label_matrix1.grid(row=8, column=3, padx=5, pady=5, columnspan=5)

# КОП�?Я МАТ� �?ЦЫ
entry_matrix1 = np.zeros((CONST_ROW, CONST_COL), dtype=object)
for i in range(CONST_ROW):  # Здесь использовано фиксированное количество строк для примера
	row_entries1 = []
	for j in range(CONST_COL):  # Здесь использовано фиксированное количество столбцов для примера
		entry1 = tk.Entry(manualInputTab, width=5)
		entry1.config(state="normal")
		entry1.insert(tk.END, '0')
		entry1.config(state="disabled")
		entry1.grid(row=i + 9, column=j + 3, padx=5, pady=5)
		entry_matrix1[i, j] = entry1

label_2 = tk.Label(manualInputTab, text="", bg=color)
label_2.grid(row=1, column=9, padx=30, pady=5)

label_alg = tk.Label(manualInputTab, text="Выбор алгоритма:", bg=colorLbl)
label_alg.grid(row=1, column=10, padx=5, pady=5)

label_split = tk.Label(manualInputTab, text="Период:", bg=colorLbl)
label_split.grid(row=1, column=11, padx=5, pady=5)

# Создаем выпадающий список с алгоритмами
algorithms = ["Венгерский алгоритм (max)", "Венгерский алгоритм (min)", "Жадный алгоритм", "Бережливый алгоритм",
			  "Бережливо-жадный алгоритм", "Жадно-бережливый алгоритм"]
algorithm_var = tk.StringVar(manualInputTab)
algorithm_var.set(algorithms[0])  # Устанавливаем значение по умолчанию
algorithm_menu = ttk.Combobox(manualInputTab, textvariable=algorithm_var, values=algorithms, width=30)
algorithm_menu.grid(row=2, column=10, padx=5, pady=5, sticky='w')

# Создаем выпадающий список с периодами
splits = [f"{i + 1}" for i in range(int(entry_cols.get()) + 1)]  # int(entry_cols.get())
# splits = ["1", "2", "3", "4", "5"]
splits_var = tk.StringVar(manualInputTab)
splits_var.set(splits[0])  # Устанавливаем значение по умолчанию
splits_menu = ttk.Combobox(manualInputTab, textvariable=splits_var, values=splits, width=2)
splits_menu.grid(row=2, column=11, padx=5, pady=5)

# Кнопка для вычисления результата
calculate_button = tk.Button(manualInputTab, text="� ассчитать", command=on_calculate_clicked, bg=colorBtn)
calculate_button.grid(row=2, column=12, padx=5, pady=5, columnspan=2)

# Создаем текстовое поле для вывода результата
result_text = tk.Text(manualInputTab, height=1, width=25, state="disabled")
result_text.grid(row=3, column=10, padx=5, pady=5, sticky='w')

label_sort = tk.Label(manualInputTab, text="Выбор № сорта:", bg=colorLbl)
label_sort.grid(row=8, column=0, padx=5, pady=5, columnspan=2)

# Создаем выпадающий список с сортами
sorts = [f"{i + 1}" for i in range(6)]  # int(entry_cols.get())
# splits = ["1", "2", "3", "4", "5"]
sorts_var = tk.StringVar(manualInputTab)
sorts_var.set(sorts[0])  # Устанавливаем значение по умолчанию
sorts_menu = ttk.Combobox(manualInputTab, textvariable=sorts_var, values=sorts, width=2)
sorts_menu.grid(row=9, column=0, padx=1, pady=5, columnspan=2)

# Кнопка для вычисления результата
calculate_button = tk.Button(manualInputTab, text="Применить", command=second_matrix, bg=colorBtn)
calculate_button.grid(row=10, column=0, padx=1, pady=5, columnspan=2)

label_alg1 = tk.Label(manualInputTab, text="Выбор алгоритма:", bg=colorLbl)
label_alg1.grid(row=8, column=10, padx=5, pady=5)

label_split = tk.Label(manualInputTab, text="Период:", bg=colorLbl)
label_split.grid(row=8, column=11, padx=5, pady=5)

# Создаем выпадающий список с алгоритмами
algorithm_var1 = tk.StringVar(manualInputTab)
algorithm_var1.set(algorithms[0])  # Устанавливаем значение по умолчанию
algorithm_menu1 = ttk.Combobox(manualInputTab, textvariable=algorithm_var1, values=algorithms, width=30)
algorithm_menu1.grid(row=9, column=10, padx=5, pady=5, sticky='w')

# Создаем выпадающий список с периодами
splits_var1 = tk.StringVar(manualInputTab)
splits_var1.set(splits[0])  # Устанавливаем значение по умолчанию
splits_menu1 = ttk.Combobox(manualInputTab, textvariable=splits_var1, values=splits, width=2)
splits_menu1.grid(row=9, column=11, padx=5, pady=5)

# Кнопка для вычисления результата
calculate_button1 = tk.Button(manualInputTab, text="� ассчитать", command=on_calculate_clicked1, bg=colorBtn)
calculate_button1.grid(row=9, column=12, padx=5, pady=5)

# Создаем текстовое поле для вывода результата
result_text1 = tk.Text(manualInputTab, height=1, width=25, state="disabled")
result_text1.grid(row=10, column=10, padx=5, pady=5, sticky='w')

# Создаем и размещаем виджеты в АВТОМАТ�?ЧЕСКОМ ВВОДЕ
label_rows1 = tk.Label(automaticInputTab, text=" ", bg=color)
label_rows1.grid(row=1, column=0, padx=200, pady=0, sticky='e')

label_rows1 = tk.Label(automaticInputTab, text="Кол-во партий и переработок:", bg=colorLbl)
label_rows1.grid(row=2, column=0, padx=0, pady=9, sticky='e')
entry_rows1 = tk.Entry(automaticInputTab, width=10)
entry_rows1.insert(tk.END, '0')
entry_rows1.grid(row=2, column=1, padx=5, pady=9, sticky='w')

entry_cols1 = entry_rows1

label_range = tk.Label(automaticInputTab, text="Диапазон сахаристости партий до переработок", bg=colorLbl)
label_range.grid(row=3, column=0, padx=0, pady=9, sticky='e')

label_min = tk.Label(automaticInputTab, text="min:", bg=colorLbl)
label_min.grid(row=4, column=0, padx=0, pady=9, sticky='e')
entry_min = tk.Entry(automaticInputTab, width=10)
entry_min.insert(tk.END, '0')
entry_min.grid(row=4, column=1, padx=5, pady=9, sticky='w')

label_max = tk.Label(automaticInputTab, text="max:", bg=colorLbl)
label_max.grid(row=4, column=2, padx=0, pady=9, sticky='e')
entry_max = tk.Entry(automaticInputTab, width=10)
entry_max.insert(tk.END, '0')
entry_max.grid(row=4, column=3, padx=5, pady=9, sticky='w')

label_range1 = tk.Label(automaticInputTab, text="Диапазон деградации:", bg=colorLbl)
label_range1.grid(row=5, column=0, padx=0, pady=9, sticky='e')

label_min1 = tk.Label(automaticInputTab, text="min:", bg=colorLbl)
label_min1.grid(row=6, column=0, padx=0, pady=9, sticky='e')
entry_min1 = tk.Entry(automaticInputTab, width=10)
entry_min1.insert(tk.END, '0')
entry_min1.grid(row=6, column=1, padx=5, pady=9, sticky='w')

label_max1 = tk.Label(automaticInputTab, text="max:", bg=colorLbl)
label_max1.grid(row=6, column=2, padx=0, pady=9, sticky='e')
entry_max1 = tk.Entry(automaticInputTab, width=10)
entry_max1.insert(tk.END, '0')
entry_max1.grid(row=6, column=3, padx=5, pady=9, sticky='w')

# label_space = tk.Label(automaticInputTab, text=" ", bg=color)
# label_space.grid(row=8, column=0, padx=0, pady=5, sticky='e')

label_inorganic = tk.Label(automaticInputTab, text="Учитывать при выходе сахара влияние неорганики", bg=colorLbl)
label_inorganic.grid(row=7, column=0, padx=0, pady=9, sticky='e')

inorganic_enabled = tk.IntVar()
checkbox_inorganic = tk.Checkbutton(automaticInputTab, variable=inorganic_enabled, command=consider_inorganic, bg=color)
checkbox_inorganic.grid(row=7, column=1, padx=5, pady=9, sticky="w")

label_sort = tk.Label(automaticInputTab, text="Выбор сорта:", bg=colorLbl, state='disabled')
label_sort.grid(row=8, column=0, padx=0, pady=9, sticky='e')

# Создаем выпадающий список с сортами
# sorts1 = [f"{i + 1}" for i in range(6)] # int(entry_cols.get())
# splits = ["1", "2", "3", "4", "5"]
sorts_var1 = tk.StringVar(automaticInputTab)
sorts_var1.set(sorts[0])  # Устанавливаем значение по умолчанию
sorts_menu1 = ttk.Combobox(automaticInputTab, textvariable=sorts_var1, values=sorts, width=2, state='disabled')
sorts_menu1.grid(row=8, column=1, padx=5, pady=9, sticky='w')

label_exp = tk.Label(automaticInputTab, text="Кол-во экспериментов:", bg=colorLbl)
label_exp.grid(row=9, column=0, padx=0, pady=9, sticky='e')
entry_exp = tk.Entry(automaticInputTab, width=10)
entry_exp.insert(tk.END, '0')
entry_exp.grid(row=9, column=1, padx=5, pady=9, sticky='w')

# Кнопка для вычисления результата
calculate_button1 = tk.Button(automaticInputTab, text="� ассчитать", command=experiments, bg=colorBtn)
calculate_button1.grid(row=10, column=0, padx=65, pady=9, columnspan=3, sticky='e')

# Запускаем главный цикл
root.mainloop()
