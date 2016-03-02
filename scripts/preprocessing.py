import os, sys, csv
import math

def roundup(var):
	return float(format(var, '.6f'))

def main(dir_path, output_dir):
	files = os.listdir(dir_path)
	for file_name in files:
		with open(dir_path + '/' + file_name, 'r') as textfile:
			new_file = open(output_dir + '/' + file_name, 'w+')
			new_list = []
			prev = 0.0
			diff = 0.0
			avg = 0.0
			num_moving_avg = 50
			volatile_avg = 0.0
			num_volatile = 10
			curr_volatility = 0.0

			for count, row in enumerate(reversed(list(csv.reader(textfile)))):
				if not count:
					row.append(prev)
				else:
					diff = roundup(float(row[5]) - float(prev))
					row.append(diff)
				
				if count<num_moving_avg:
					avg = roundup((count * avg + float(row[5]))/ (count + 1))
				else:
					avg = roundup((num_moving_avg * avg + float(row[5]) - float(new_list[count - num_moving_avg][5])) / (num_moving_avg)) 

				prev = float(row[5])

				if count < num_volatile:
					volatile_avg = roundup((count * volatile_avg + float(row[5]))/ (count + 1))
				else:
					volatile_avg = roundup((num_volatile * volatile_avg + float(row[5]) - float(new_list[count - num_volatile][5])) / (num_volatile))

				if count:
					loop_count = min(count, num_volatile)
				
					for i in range(loop_count):
						curr_volatility += math.pow((float(row[5]) - volatile_avg), 2)

					curr_volatility = roundup(math.sqrt(curr_volatility / (loop_count))) 
				
				row.append(avg)
				row.append(curr_volatility)
				new_list.append(row)
				curr_volatility = 0.0

			writer = csv.writer(new_file)
			writer.writerows(new_list)
			new_file.close()
		textfile.close()

if __name__ == '__main__':
	main(str(sys.argv[1]), str(sys.argv[2]))