import csv
import sys

def text_to_csv(text_file, csv_file):
	with open(text_file, encoding="utf-8") as tf:
		line = tf.readline()
		cf = open(csv_file, 'w', newline='')
		filewriter = csv.writer(cf, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		filewriter.writerow(['label', 'sentence'])

		while line:
			new_line = line.rstrip('\n')
			tokens = new_line.split("\t")
			sentence = tokens[1]
			mood = tokens[2]
			intensity = float(tokens[3])

			base_label = 0
			if mood == "anger":
				base_label = 1
			elif mood == "fear":
				base_label = 4
			elif mood == "joy":
				base_label = 7
			elif mood == "sadness":
				base_label = 10

			intensity_label = 0
			if intensity <= 0.33:
				intensity_label = 0
			elif intensity > 0.33 and intensity <= 0.67:
				intensity_label = 1
			elif intensity > 0.67 and intensity <= 1:
				intensity_label = 2

			label = base_label + intensity_label

			filewriter.writerow([label, sentence])

			line = tf.readline()

def main():
	text_to_csv(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
	main()