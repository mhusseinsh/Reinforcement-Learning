import numpy as np

def evaluate(exam):
	global task_pointer
	global horizon_check
	global prob_of_succes
	global reward
	global current_value
	global next_value
	value = 0
	p = np.round(np.random.random_sample(),5)

	if (exam[task_pointer][2] != "S"):
		prob_of_succes = exam[task_pointer][0]
		reward = exam[task_pointer][1]
		current_value = exam[task_pointer][3]
		next_value = 0
		if task_pointer != 3:
			next_value = exam[task_pointer+1][3]
		
		if (prob_of_succes >= p):
			value = prob_of_succes * (reward + next_value) + (1-prob_of_succes) * current_value
			exam[task_pointer] = (prob_of_succes, reward, "S", value)
			
			#print ("i successed in ",x,"question ",i,"horizon_check")
			#print (exam)
			if task_pointer == 3:
				value = exam[0][3] + exam[1][3] + exam[2][3] + exam[3][3]
			#	print ("early success ",exam)
				return value
			task_pointer += 1
			horizon_check += 1
			evaluate(exam)
		if horizon_check < 6:
			#print("i",i)
			#print ("i failed in ",x,"question ",i,"horizon_check")
			#print (exam)
			horizon_check += 1
			evaluate(exam)
	count_fail = 0
	for j in range(4):
		if exam[j][2] == "F":
			count_fail += 1
	if count_fail == 4:

		value = prob_of_succes * (reward + next_value) + (1-prob_of_succes) * (-10 + current_value)
	else:
		value = exam[0][3] + exam[1][3] + exam[2][3] + exam[3][3]
	
	return value

def evaluate_stationary(exam):
	global task_pointer
	global prob_of_succes
	global reward
	global current_value
	global next_value
	value = 0
	p = np.round(np.random.random_sample(),5)

	if (exam[task_pointer][2] != "S"):
		prob_of_succes = exam[task_pointer][0]
		reward = exam[task_pointer][1]
		current_value = exam[task_pointer][3]
		next_value = 0
		if task_pointer != 3:
			next_value = exam[task_pointer + 1][3]
		
		if (prob_of_succes >= p):
			value = prob_of_succes * (reward + next_value) + (1-prob_of_succes) * current_value
			exam[task_pointer] = (prob_of_succes, reward, "S", value)
			if task_pointer == 3:
				value = exam[0][3] + exam[1][3] + exam[2][3] + exam[3][3]
				return value
			task_pointer += 1
			evaluate(exam)
		else:
			evaluate(exam)
	count_fail = 0
	for j in range(4):
		if exam[j][2] == "F":
			count_fail += 1
	if count_fail == 4:

		value = prob_of_succes * (reward + next_value) + (1-prob_of_succes) * (-10 + current_value)
	else:
		value = exam[0][3] + exam[1][3] + exam[2][3] + exam[3][3]
	
	return value

if __name__ == "__main__":
	exam_a = [(0.8, 1, "F", 0), (0.5, 2, "F", 0), (0.3, 3, "F", 0), (0.1, 4, "F", 0)]
	exam_b = [(0.1, 4, "F", 0), (0.3, 3, "F", 0), (0.5, 2, "F", 0), (0.8, 1, "F", 0)]
	exam_c = [(0.1, 4, "F", 0), (0.5, 2, "F", 0), (0.8, 1, "F", 0), (0.3, 3, "F", 0)]
	student_t = [(0.5, 4, "F", 0), (0.0, 3, "F", 0), (0.0, 2, "F", 0), (0.0, 1, "F", 0)]
	task_pointer = 0
	horizon_check = 1
	prob_of_succes = 0
	reward = 0
	current_value = 0
	next_value = 0		
	print("Value of policy a:",evaluate(exam_a))
	task_pointer = 0
	horizon_check = 1
	prob_of_succes = 0
	reward = 0
	current_value = 0
	next_value = 0		
	print("Value of policy b:",evaluate(exam_b))
	task_pointer = 0
	horizon_check = 1
	prob_of_succes = 0
	reward = 0
	current_value = 0
	next_value = 0		
	print("Value of policy c:",evaluate_stationary(exam_c))
	task_pointer = 0
	horizon_check = 1
	prob_of_succes = 0
	reward = 0
	current_value = 0
	next_value = 0		
	print("Value of student t:",evaluate(student_t))