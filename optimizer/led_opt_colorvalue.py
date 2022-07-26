import sys
import itertools
import numpy
from scipy.optimize import minimize
import math

import illuminant
from common import common

# primaryLEDSpectrum = [[0 for j in range(71)] for i in range(18)]
# primaryLED= [0 for i in range(len(primaryLEDSpectrum))]
LEDCombinations = list(itertools.combinations((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18),3))
LED_COUNT = common.LED_COUNT

# debug = False

# if debug:
#     f = open('led.combinations.txt','w')
#     for v in LEDCombinations:
#         f.write(str(v)+'\n')
#     f.close()

def _func(x, *a):
    c = a[2]
    p = x
    e = 0.0

    for i in range(LED_COUNT):
        e = e + p[i] * c[i]

    return e

def _cons1(x):
    return x

def _cons2(x):
    return 1.0 - x

def _cons3(x,*a):
    XYZ = a[0]
    XYZn = a[1]
    
    X1 = XYZ[0]
    X2 = 0.0
    for i in range(LED_COUNT):
        X2 = X2 + x[i] * XYZn[i][0]

    return X1 - X2

def _cons4(x,*a):
    XYZ = a[0]
    XYZn = a[1]

    Y1 = XYZ[1]
    Y2 = 0.0
    for i in range(LED_COUNT):
        Y2 = Y2 + x[i] * XYZn[i][1]

    return Y1 - Y2

def _cons5(x,*a):
    XYZ = a[0]
    XYZn = a[1]

    Z1 = XYZ[2]
    Z2 = 0.0
    for i in range(LED_COUNT):
        Z2 = Z2 + x[i] * XYZn[i][2]

    return Z1 - Z2

def _optimize(XYZ, XYZn, led):

    p = [0.1 for i in range(LED_COUNT)]

    c = [0.0 for i in range(LED_COUNT)]
    for i in range(18):
        c[i] = XYZn[i][1]
    m = max(c)
    for i in range(LED_COUNT):
        c[i] = c[i] / m

    #minimizer SLSQP method
    cons = (
        {'type': 'ineq', 'fun': _cons1},
        {'type': 'ineq', 'fun': _cons2},
        {'type': 'eq', 'fun': _cons3, 'args': (XYZ, XYZn)},
        {'type': 'eq', 'fun': _cons4, 'args': (XYZ, XYZn)},
        {'type': 'eq', 'fun': _cons5, 'args': (XYZ, XYZn)},
    )
    # bounds = ((0.0, 1.0), (0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0))

    res = minimize(_func, x0=p, constraints=cons, args=(XYZ, XYZn, c), method='SLSQP')

    if res.success == False:
        return None

    p = res.x

    # Ps = max(p)
    # for i in range(18):
    #     p[i] = p[i] / Ps

    resIntensity = [0 for i in range(71)]

    for i in range(71):
        for j in range(LED_COUNT):
            resIntensity[i] += p[j] * led[j][i]

    resluminance = 0
    for i in range(71):
        resluminance += resIntensity[i] * illuminant.color_matching_function_Y[i]
    resluminance = resluminance * 5 * 683

    X,Y,Z = illuminant.spectrum2XYZ(resIntensity)
    
    l,x,y = illuminant.XYZ2lxy(X, Y, Z)
    ct = illuminant.cct(X, Y, Z)
    lab = illuminant.XYZ2LAB([X,Y,Z])
    # refIntensity = sp

    p2 = [0 for i in range(LED_COUNT)]
    for i in range(LED_COUNT):
        p2[i] = float(p[i])

    return {"result":{"intensity":resIntensity, "p":p2, "luminance":resluminance, "XYZ":[X,Y,Z], "xy": [l,x,y], "ct": ct, "lab":lab}}

def inside_polygon(x, y, points):
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# def init():
#     pass

def optimize(X, Y, Z):
    import led
    XYZn = led.XYZn

    n = 0
    for i in range(len(LEDCombinations)):
        ch = LEDCombinations[i]
        XYZ1 = XYZn[ch[0]-1]
        XYZ2 = XYZn[ch[1]-1]
        XYZ3 = XYZn[ch[2]-1]
        xy1 = (XYZ1[0]/(XYZ1[0]+XYZ1[1]+XYZ1[2]), XYZ1[1]/(XYZ1[0]+XYZ1[1]+XYZ1[2]))
        xy2 = (XYZ2[0]/(XYZ2[0]+XYZ2[1]+XYZ2[2]), XYZ2[1]/(XYZ2[0]+XYZ2[1]+XYZ2[2]))
        xy3 = (XYZ3[0]/(XYZ3[0]+XYZ3[1]+XYZ3[2]), XYZ3[1]/(XYZ3[0]+XYZ3[1]+XYZ3[2]))
        polygon = [xy1,xy2,xy3]
        target = [X/(X+Y+Z), Y/(X+Y+Z)]
        if(inside_polygon(target[0],target[1], polygon)):
            n = n + 1

    if n == 0:
        return None

    # XYZn = []

    # for i in range(18):
    #     XYZn.append(primaryLED[i][1])

    XYZ = [X,Y,Z]

    res = _optimize(XYZ, XYZn, led.LED)

    if res == None:
        return None

    # f = open('led_opt.solver.txt','w')
    # for i in range(71):
    #     f.write(str(res["cal"]["intensity"][i])+'\n')
    # f.close()

    return res

def optimizeXYZ(X,Y,Z):

    X = float(X)
    Y = float(Y)
    Z = float(Z)

    return optimize(X,Y,Z)

def optimizexyY(x,y,Y):

    x = float(x)
    y = float(y)
    Y = float(Y)

    X, Y, Z = illuminant.xyY2XYZ(x,y,Y)

    return optimize(X, Y, Z)

def optimizelab(l,a,b, white_luminance):
    
    l = float(l)
    a = float(a)
    b = float(b)

    X, Y, Z = illuminant.LAB2XYZ([l,a,b])
    
    X = X / 100.0 * white_luminance
    Y = Y / 100.0 * white_luminance
    Z = Z / 100.0 * white_luminance
    
    return optimize(X,Y,Z)

def optimizeRGB(r,g,b):
    
    r = float(r)
    g = float(g)
    b = float(b)

    X,Y,Z = illuminant.RGB2XYZ(r,g,b)

    X = X
    Y = Y
    Z = Z

    return optimize(X,Y,Z)

if __name__ == '__main__':
    import led
    led.configure("low")

    # XYZ_E = [100,100,100]
    # XYZ_D65 = [88,92,100]
    # XYZ_D55 = [95,100,92]
    # XYZ_D50 = [96,100,83]
    # XYZ_sRGB_R = [35,18,1]
    # XYZ_sRGB_G = [28,56,9]
    # XYZ_sRGB_B = [14,6,73]

    XYZ1 = [100,100,100]
    X,Y,Z = XYZ1
    L,x,y = illuminant.XYZ2lxy(X,Y,Z)
    # print x
    # print y
    # res = optimizeXYZ(X,Y,Z)

    xy_SHV_R = [0.708, 0.292] #SHV Red
    xy_SHV_G = [0.170, 0.797]  #SHV Green
    xy_SHV_B = [0.131, 0.046]  #SHV Blue
    xy_sRGB_R = [0.64, 0.33] #sRGB Red
    xy_sRGB_G = [0.3, 0.6]  #sRGB Green
    xy_sRGB_B = [0.15, 0.06]  #sRGB Blue
    xy_D65 = [0.3127, 0.329]  #D65
    xy_E = [0.3333, 0.3333] #E
    xy_adobeRGB_R = [0.64, 0.33] #adobeRGB Red
    xy_adobeRGB_G = [0.21, 0.71]  #adobeRGB Green
    xy_adobeRGB_B = [0.15, 0.06]  #adobeRGB Blue

    xy = xy_E

    l = 10

    res = optimizexyY(xy[0],xy[1],l)
    # res = optimizelab(20, -50, 0, 500)

    if res == None:
        print "parameter error."
        sys.exit(-1)

    sp = res["result"]["intensity"]
    p = res["result"]["p"]
    XYZ = illuminant.spectrum2XYZ(sp)
    X=XYZ[0]
    Y=XYZ[1]
    Z=XYZ[2]
    xyY = illuminant.XYZ2lxy(X,Y,Z)
    x = xyY[1]
    y = xyY[2]
    print "x,y = %f, %f" % (x,y)
    
    lab = illuminant.XYZ2LAB(XYZ)
    l = int(round(lab[0],0))
    a = int(round(lab[1],0))
    b = int(round(lab[2],0))
    
    luminance = illuminant.Luminance(sp)

    ct = illuminant.cct(X,Y,Z)

    print "factor"
    # for i in range(18):
    #     print "p%d = %f" % (i+1, p[i])
    print p

    print "XYZ = %f, %f, %f" % (X, Y, Z)
    print "xy = %f, %f" % (x, y)
    print "lab = %d, %d, %d" % (l,a,b)
    print "luminance = %f cd/m2" % luminance
    print "T = %f K" % ct

    p2 = [0]*LED_COUNT

    for i in range(LED_COUNT):
        p2[i] = int(p[i] * 4000)

    # print p2
    # import led_controller
    # import serial
    # ser = serial.Serial('/dev/ttyS2',115200)
    # led_controller.sendPwmValue(ser, p2)
 