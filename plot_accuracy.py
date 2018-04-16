import os
import pdb
import matplotlib.pyplot as plt

# pdb.set_trace()


cur_file="result/"
str_file = os.listdir(cur_file);
file_list = [int(file) for file in str_file]
file_list = sorted(file_list);
sample_train_accuracy=[];
sample_test_accuracy=[];
soft_train_accuracy=[];
soft_test_accuracy=[];
for file in file_list:
	cur_path=cur_file+str(file)+'/score.txt'
	F = open(cur_path,"rb")
	for cur_line in F:
		line = cur_line.decode("utf-8")
		str_score = line[line.find(":")+2: line.find("%")-1]
		score = int(str_score)

		if("Sample Train Accuracy" in line):
			sample_train_accuracy.append(score);
		elif("Sample Test Accuracy" in line):
			sample_test_accuracy.append(score);
		elif("soft_Train Accuracy" in line):
			soft_train_accuracy.append(score);
		elif("soft_Test Accuracy" in line):
			soft_test_accuracy.append(score);
		else:
			print("None")

# print(sample_train_accuracy)
# print(sample_test_accuracy)
plt.ylim(0,100)
plt.plot(sample_train_accuracy)
plt.plot(sample_test_accuracy)
plt.show()


