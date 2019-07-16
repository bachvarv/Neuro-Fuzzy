import numpy as np

# file = open("test2.txt", "r")
#
# lines = file.readlines()
# oneline = lines[0].split("\t")
# size = len(oneline) - 1
# anz = len(lines)
# xVal = np.zeros(shape=(anz, size), dtype=np.float)
# # xVal = []
# yVal = np.zeros(shape=(anz, 1), dtype=np.float)
# lineInd = 0
# file.close()
#
# file = open("test2.txt", "r")
#
# if file.mode == "r":
#     line = file.readline()
#     while(line != ""):
#         newline = line.replace("\n", "")
#         items = newline.split("\t")
#
#         arr = np.array(items).astype(np.float)
#         xVal[lineInd] = arr[:-1]
#         yVal[lineInd] = arr[-1]
#
#         line = file.readline()
#         lineInd += 1
#
# print(xVal)
# print(yVal)
# file.close()


def readFile(path):
    with open(path, "r") as file:
        xArr, yArr = arrays(file)


    with open(path, "r") as file:
        lineInd = 0
        if file.mode == "r":
            line = file.readline()
            while (line != ""):
                newline = line.replace("\n", "")
                items = newline.split("\t")

                arr = np.array(items).astype(np.float64)
                xArr[lineInd] = arr[:-1]
                yArr[lineInd] = arr[-1]

                line = file.readline()
                lineInd += 1

    return xArr, yArr


def arrays(file):
    lines = file.readlines()
    oneline = lines[0].split("\t")
    column = len(oneline) - 1
    rows = len(lines)
    xArr = np.zeros(shape=(rows, column), dtype=np.float)
    yArr = np.zeros(shape=(rows, 1), dtype=np.float)
    return xArr, yArr


def range_one_input(path, index):
    val_range = np.zeros(shape=(2), dtype=np.float)
    lowest = np.math.inf
    highest = -np.math.inf
    lineInd = 0
    with open(path, 'r') as file:
        if file.mode == 'r':
            line = file.readline()
            while(line != ""):
                newline = line.replace("\n", "")
                items = newline.split("\t")

                arr = np.array(items).astype(np.float64)
                if(lowest > arr[index]):
                    lowest = arr[index]
                if(highest < arr[index]):
                    highest = arr[index]
                line = file.readline()

        # NOTE: The amount of wiggle room for the value changes the error rate drastically

        res = 0
        if lowest > -10 and lowest < 10:
            res = 1e-2
        elif lowest >= -100 and lowest > -1000:
            res = 2*abs(lowest) / 100
        else:
            res = 2*(abs(lowest) / 1000)


        val_range[0] = lowest - res
        val_range[1] = highest + res
    return val_range


def createFile(xArr, yArr, name):
    file = open(name, "w+")
    yInd = 0
    for i in xArr:
        for f in i:
            file.write(str(f) + "\t")
        # for k in yArr:
        file.write(str(yArr[0][yInd]) + "\n")
        yInd += 1

    file.close()

# xArr, yArr = readFile("test2.txt")

# print(xArr, yArr)
#
# trainData = xArr[:5]
# trainYData = yArr[:5]

# xArr, yArr = readFile("data.txt")

# size = (3/4)*len(xArr)

# print(int(size))

array = np.ones(shape=(2,2), dtype=float)

# print(array)

converted = np.ndarray(shape=(len(array), len(array[0])) ,buffer=array)
# xVal = xArr[:size]
#
# print(xVal)
#
# testData = yArr[5:]