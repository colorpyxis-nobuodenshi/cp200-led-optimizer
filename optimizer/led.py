import csv

import illuminant

from common import common

LED = []
linearLUT = []
XYZn = []
LED_COUNT = common.LED_COUNT

def configure(mode="low"):
    global LED
    global linearLUT
    global XYZn

    led = [[0 for j in range(71)] for i in range(LED_COUNT)]

    linearLUT = [[0 for j in range(4096)] for i in range(LED_COUNT)]

    leddata_file = "led.txt"
    # lineardata_file = "linearLUT.txt"
    # if mode == "low":
    #     leddata_file = "led1khz.txt"
    #     lineardata_file = "linearLUT.txt"
    # elif mode == "high":
    #     leddata_file = "led20khz.txt"
    #     lineardata_file = "linearLUT.txt"

    f = open(leddata_file, "r")

    reader = csv.reader(f)
    # header = next(reader)

    j = 0
    for row in reader:
        for k in range(71):
            led[j][k] = float(row[k+1])
            
        j+=1

    LED = led

    f.close()
    
    # f = open(lineardata_file,"r")
    # reader = csv.reader(f)
    # header = next(reader)
    # j=0
    # for row in reader:
    #     for i in range(LED_COUNT):
    #         linearLUT[i][j]=int(row[i+1]) if int(row[i+1]) >= 0 else 0
    #     j = j + 1
    # f.close()

    XYZ = []
    for i in range(LED_COUNT):
        XYZ.append(illuminant.spectrum2XYZ(led[i]))

    # f = open("led.primary.XYZ.txt","w")
    # for i in range(LED_COUNT):
    #     f.write(str(XYZ[i][0])+","+str(XYZ[i][1])+","+str(XYZ[i][2])+"\n")

    # f.close()
    
    XYZn = XYZ

def synthesize(p):

    res = [0 for i in range(71)]

    for i in range(71):
        for j in range(LED_COUNT):
            res[i] += p[j] * LED[j][i]
    return res


def correct_linear(p):
    res = [0.0 for i in range(LED_COUNT)]

    for i in range(LED_COUNT):
        val = linearLUT[i][p[i]]
        if val > 4095:
            val = 4095
        res[i] = val
    return res

def test():
    configure()

    p = [1 for j in range(LED_COUNT)]

    sp = synthesize(p)

    f2 = open("res.txt","w")
    for s in sp:
        f2.write(str(s)+"\n")
    f2.close()

if __name__ == "__main__":
    test()