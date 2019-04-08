import jsonlines

question = "What’s the tweet’s emotion, angry or fearful or joyful or sad, and what’s the intensity level, slightly or fairly or extremely?"

mood = {'anger':'angry', 'fear':'fearful', 'joy':'joyful', 'sadness':'sad'}

# train files now, also need to dump test file later
filepath = ["train/anger_train.txt", "train/fear_train.txt", "train/joy_train.txt", "train/sadness_train.txt"]

#just for display of jsonl file
displaynum = 10

# This should be '.data/mood_detection_dataset/val.jsonl'
# And we need to run commands: mkdir -p .data/my_custom_dataset/ first to create .data folder
output_path = 'train.jsonl' 

def label(its, noun):
    if its >= 0 and its < 0.33:
        return "slight " + mood[noun]
    elif its >= 0.33 and its < 0.67:
        return "fairly " + mood[noun]
    else:
        return "extremely " + mood[noun]

def to_jsonl():
    for file in filepath:
        num = 0
        with open(file) as f:
            with jsonlines.open(output_path, mode='a') as writer:
                for line in f.readlines():
                    l = line.split()
                    context = ' '.join(l[1:-2])
                    answer = label(float(l[-1]), l[-2])
                    writer.write({"context": context, "question": question, "answer": answer})
                    num += 1
        print(f"this file contains {num} lines")

def display_jsonl():
    with jsonlines.open('output.jsonl', mode='r') as reader:
        i = 0
        for obj in reader:
            if i <= displaynum:
                print(obj)
                i += 1
            else: 
                break

if __name__ == '__main__':
    to_jsonl()