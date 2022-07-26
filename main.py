import sys, os

from common import common
from optimizer import led, led_opt, illuminant
# Tc = [6500]
Tc = [2856, 3500, 4300, 5003, 5503, 6504, 7504]
# Tc = [2800, 3500, 4300, 5000, 5500, 6500, 7500]
# lumminance = [100,100,100,100,100,100,100]
lumminance = [15000,15000,15000,15000,15000,15000,15000]
power = [100]

# kinds = [6500]

led.configure()

ledIntensities = {}

if os.path.exists('res') == False:
    os.mkdir('res')

for i in range(len(Tc)):
    t = Tc[i]
    b = lumminance[i]
    sp = illuminant.daylight(t)
    # print sp

    f = open('res/' + str(t)[:-2] + '00.daylight.txt', 'w')
    for i in range(71):
        f.write(str(int(sp[i]))+"\n")
    f.close()
    
    for j in range(len(power)):        
        b2 = float(b) * float(power[j]) / 100.0
        name = str(t)[:-2] + '00_' + str(power[j])
        
        res = led_opt.optimize(sp, b2, led.LED)

        if res == None:
            print("error.")
            continue

        intensity = res["out"]["intensity"]
        p = res["out"]["p"]
        XYZ = res["out"]["XYZ"]
        xy = res["out"]["lxy"]
        cct = res["out"]["ct"]
        luminance = res["out"]["luminance"]

        refIntensity = res["ref"]["intensity"]
        XYZ2 = res["ref"]["XYZ"]
        xy2 = res["ref"]["lxy"]
        cct2 = res["ref"]["ct"]
        luminance2 = res["ref"]["luminance"]

        sr2l=0.0
        sr2l=res["result"]["sr2l"]

        ledIntensities[name] = p

        # print "factor"
        # print p
        # for i in range(18):
        #     print "p%d = %f" % (i+1, p[i])

        X = XYZ[0]
        Y = XYZ[1]
        Z = XYZ[2]
        x = xy[1]
        y = xy[2]

        print("calclation .")
        # print "XYZ = %f, %f, %f" % (X, Y, Z)
        print("xy = %f, %f" % (x, y))
        # print("luminance : %f cd/m2" % luminance)
        print("luminance : %f lx" % luminance)
        print("T : %f K" % cct)

        X2 = XYZ2[0]
        Y2 = XYZ2[1]
        Z2 = XYZ2[2]
        x2 = xy2[1]
        y2 = xy2[2]

        print("reference .")
        # print "XYZ = %f, %f, %f" % (X2, Y2, Z2)
        print("xy = %f, %f" % (x2, y2))
        # print "luminance : %f cd/m2" % luminance2
        print("T : %f K" % cct2)
        print("sr2l = %f" % (sr2l))
        # if "light" in res:
        #     lab = res["out"]["lab"]
        #     lab2 = res["ref"]["lab"]
        #     sr2 = res["result"]["sr2"]
        #     sr2l = res["result"]["sr2l"]
        #     delta_e = res["result"]["de00"]
        #     print ""
        #     print "lab1 = %f,%f,%f" % (lab[0],lab[1],lab[2])
        #     print "lab2 = %f,%f,%f" % (lab2[0],lab2[1],lab2[2])
        #     print "delta e (CIE2000) = %f" % delta_e
        #     print "SR2 = %f" % sr2
        #     print "SR2L = %f" % sr2l

        debug = True
        if debug:
            if os.path.exists('res') == False:
                os.mkdir('res')
            
            f = open('res/'+ name + '.res.txt', 'w')
            for i in range(71):
                f.write(str(intensity[i])+"\n")
            f.close()

            f = open('res/' + name + 'ref.txt', 'w')
            for i in range(71):
                f.write(str(refIntensity[i])+"\n")
            f.close()
            # f = open('res/'+ name + '.ref.txt', 'w')
            # for i in range(71):
            #     f.write(str(refIntensity[i])+"\n")
            # f.close()

            # f = open('res/'+ key + '.p.txt', 'w')
            # for i in range(common.LED_COUNT):
            #     f.write(str(p[i])+"\n")
            # f.close()

keylist = ledIntensities.keys()
keylist.sort()
f = open('res/p.txt', 'w')
for n in keylist:
    p = ledIntensities[n]
    n2 = n.replace('_',':')
    f.write(n2)
    for i in range(common.LED_COUNT):
        f.write(',')
        f.write(str(int(p[i] * 1023)))
    f.write('\n')
f.close()
