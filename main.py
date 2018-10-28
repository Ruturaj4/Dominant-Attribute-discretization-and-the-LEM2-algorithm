# Author - Ruturaj Kiran Vaidya
# This is the main file

import re

# For tab completion
import readline
readline.parse_and_bind("tab: complete")


def readFile():
    f = input("Please enter the name of the input file: ")
    print("You entered: " + f)
    inputFile = []
    with open(f, "r") as filename:
        for line in filename:
            # Remove all the comments
            line = re.sub(re.compile("!.*?\n"), "", line)
            if line != "":
                inputFile.append(line)
    return inputFile

#def scanFile(inputFile):
    

def main():
    inputFile = readFile()
    # Remove the <> line (which will be first 1 in this case)
    inputFile = inputFile[1:]
    # Remove all the blank lines
    inputFile = "".join(str(x) for x in inputFile)
    print(inputFile)

if __name__=="__main__":
    main()
