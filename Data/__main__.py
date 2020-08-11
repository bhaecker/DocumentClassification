from .Preprocessing import make_split, split

def __main__():
    try:
        sum = split[0]+split[1]+split[2]
        if sum != 100:
            print("Please provide a split which adds up to 100")
        else:
            make_split(split)
    except:
        print("Provide a triple")

if __name__ == "__main__":
    __main__()