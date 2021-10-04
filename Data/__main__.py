#from .Preprocessing import make_split, split
from .Preprocessing_include_names import make_split, split

def __main__():
    sum = split[0]+split[1]+split[2]
    if sum != 100:
        print("Please provide a triple (in list form) which adds up to 100 and represents the desired split in train/test/unseen")
    else:
        make_split(split)

if __name__ == "__main__":
    __main__()