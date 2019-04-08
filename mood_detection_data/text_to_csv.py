import csv
import sys

mood = {'anger':'angry', 'fear':'fearful', 'joy':'joyful', 'sadness':'sad'}
def label(its, noun):
    if its >= 0 and its < 0.33:
        return "slightly " + mood[noun]
    elif its >= 0.33 and its < 0.67:
        return "fairly " + mood[noun]
    else:
        return "extremely " + mood[noun]

def text_to_csv(text_file, csv_file):
    with open(text_file, encoding="utf-8") as tf:
        line = tf.readline()
        with open(csv_file, 'w', newline='') as cf:
            filewriter = csv.writer(cf, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['label', 'sentence'])

            while line:
                new_line = line.rstrip('\n')
                tokens = new_line.split("\t")
                sentence = tokens[1]
                mood = tokens[2]
                intensity = float(tokens[3])
    
                filewriter.writerow([label(intensity,mood), sentence])
        
                line = tf.readline()


def main():
	text_to_csv(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
	main()
