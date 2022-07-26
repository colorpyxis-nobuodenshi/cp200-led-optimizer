# -*- coding: utf-8 -*-
import numpy
from scipy.optimize import minimize
import time
import math

import illuminant
from common import common

solve_factor = 0.0
LED_COUNT = common.LED_COUNT

def _func1(x, *a):
    target = numpy.array(a[0])
    cal = numpy.array(a[0])
    base = numpy.array(a[1])
    Ps = a[2]
    p = numpy.array(x)
    e2 = 0.0
    sigmaCal = 0.0
    N = 71
    YN = numpy.mean(target)
    Yv = 0.0#1.0
    v=sum(illuminant.color_matching_function_Y)
    for i in range(N):
        Yv = Yv + illuminant.color_matching_function_Y[i] / v * target[i]
    Yv = Yv / YN

    for i in range(71):
        sigmaCal = 0.0
        for j in range(LED_COUNT):
            sigmaCal = sigmaCal + p[j] * base[i][j] / Ps
        e2 = e2 + pow((target[i] - Ps * sigmaCal)/(YN*Yv), 2)

        cal[i] = Ps * sigmaCal

    e2 = e2 / N

    XYZ1 = illuminant.spectrum2XYZ(target)
    XYZ2 = illuminant.spectrum2XYZ(cal)
    # xy1 = illuminant.XYZ2lxy(XYZ1[0], XYZ1[1], XYZ1[2])
    # xy2 = illuminant.XYZ2lxy(XYZ2[0], XYZ2[1], XYZ2[2])

    e22 = pow(XYZ1[0] - XYZ2[0], 2) + pow(XYZ1[1] - XYZ2[1], 2) + pow(XYZ1[2] - XYZ2[2], 2)
    # e22 = pow(xy1[1] - xy2[1], 2) + pow(xy1[2] - xy2[2], 2)

    factor = 0.25
    return (1.0 - factor) * e2 + factor * e22 * 0.001
    # return e2

def _cons1(x):
    return x

def _cons2(x):
    return 1.0 - x

def _cons3(x, *a):
    ref = a[0]
    base = a[1]
    Ps = a[2]

    XYZ1 = illuminant.spectrum2XYZ(ref)
    cal = numpy.zeros(71)
    for i in range(71):
        sigmaCal = 0.0
        for j in range(LED_COUNT):
            sigmaCal = sigmaCal + x[j] * base[i][j] / Ps
        cal[i] = sigmaCal * Ps

    XYZ2 = illuminant.spectrum2XYZ(cal)

    return XYZ2[1] - XYZ1[1]

def optimizeRelative(targetSpectrum, outputLevel, led):
    ref = targetSpectrum

    a = [[0 for i in range(LED_COUNT)] for j in range(71)]

    for i in range(71):
        for j in range(LED_COUNT):
            a[i][j] = led[j][i]

    #lsq
    a = numpy.array(a)
    b = numpy.array(ref)

    Q,R = numpy.linalg.qr(a)
    t = numpy.dot(Q.T, b)
    p = numpy.linalg.solve(R, t)
  
    Ps = max(p)

    #minimizer SLSQP method
    cons = (
        {'type': 'ineq', 'fun': _cons1},
        # {'type': 'ineq', 'fun': _cons2},
    )

    res = minimize(_func1, x0=p, constraints=cons, bounds=None, args=(ref, a, Ps), method='SLSQP')

    if res.success == False:
        return None

    p = res.x

    Ps = max(p)
    for i in range(LED_COUNT):
        p[i] = p[i] / Ps * outputLevel / 100.0

    resIntensity = [0.0 for i in range(71)]

    for i in range(71):
        for j in range(LED_COUNT):
            resIntensity[i] += p[j] * led[j][i]

    resluminance = 0
    for i in range(71):
        resluminance += resIntensity[i] * illuminant.color_matching_function_Y[i]
    resluminance = resluminance * 5 * 683

    resXYZ = illuminant.spectrum2XYZ(resIntensity)
    resxyY = illuminant.XYZ2lxy(resXYZ[0],resXYZ[1],resXYZ[2])
    resCT = illuminant.cct(resXYZ[0], resXYZ[1], resXYZ[2])

    refIntensity = [0 for i in range(71)]
    for i in range(71):
        refIntensity[i] = ref[i] / Ps * outputLevel / 100.0

    refluminance = illuminant.Luminance(refIntensity)
    refXYZ = illuminant.spectrum2XYZ(refIntensity)
    refxyY = illuminant.XYZ2lxy(refXYZ[0],refXYZ[1],refXYZ[2])
    refCT = illuminant.cct(refXYZ[0], refXYZ[1], refXYZ[2])
    
    p2 = [0 for i in range(LED_COUNT)]
    for i in range(LED_COUNT):
        p2[i] = float(p[i])
    
    sr2 = illuminant.SR2(refIntensity,resIntensity)
    sr2l = illuminant.SR2L(refIntensity,resIntensity)
    
    return {"result":{"p":p2, "ps":Ps, "sr2":sr2, "sr2l":sr2l},"out":{"intensity":resIntensity, "p":p2, "luminance":resluminance, "XYZ":resXYZ, "lxy": resxyY, "ct": resCT},"ref":{"intensity":refIntensity,"luminance":refluminance, "XYZ":refXYZ, "lxy":refxyY, "ct":refCT}}

def optimize(targetSpectrum, brightness, led):
    ref = targetSpectrum
    radiance = 0
    targetluminance = 0
    luminance = brightness

    for i in range(71):
        radiance += targetSpectrum[i]
        targetluminance += targetSpectrum[i] * illuminant.color_matching_function_Y[i]
    
    targetluminance = targetluminance * 5 * 683

    factor = radiance / targetluminance * luminance / radiance

    for i in range(71):
        ref[i] = targetSpectrum[i] * factor

    a = [[0 for i in range(LED_COUNT)] for j in range(71)]

    for i in range(71):
        for j in range(LED_COUNT):
            a[i][j] = led[j][i]

    #lsq
    a = numpy.array(a)
    b = numpy.array(ref)
    
    Q,R = numpy.linalg.qr(a)
    t = numpy.dot(Q.T, b)
    
    p = numpy.linalg.solve(R, t)

    Ps = max(p)

    #minimizer SLSQP method
    cons = (
        {'type': 'ineq', 'fun': _cons1},
        {'type': 'ineq', 'fun': _cons2},
        # {'type': 'ineq', 'fun': _cons3, 'args': (ref, a, Ps)},
    )

    res = minimize(_func1, x0=p, constraints=cons, args=(ref, a, Ps), method='SLSQP')

    if res.success == False:
        return None
    
    #p = res.x

    resIntensity = [0.0 for i in range(71)]

    for i in range(71):
        for j in range(LED_COUNT):
            resIntensity[i] += p[j] * led[j][i]

    resluminance = 0
    for i in range(71):
        resluminance += resIntensity[i] * illuminant.color_matching_function_Y[i] / 0.00146
    resluminance = resluminance * 5 * 683

    resXYZ = illuminant.spectrum2XYZ(resIntensity)
    resxyY = illuminant.XYZ2lxy(resXYZ[0],resXYZ[1],resXYZ[2])
    resCT = illuminant.cct(resXYZ[0], resXYZ[1], resXYZ[2])

    refIntensity = [0 for i in range(71)]
    for i in range(71):
        refIntensity[i] = ref[i]

    refluminance = illuminant.Luminance(refIntensity)
    refXYZ = illuminant.spectrum2XYZ(refIntensity)
    refxyY = illuminant.XYZ2lxy(refXYZ[0],refXYZ[1],refXYZ[2])
    refCT = illuminant.cct(refXYZ[0], refXYZ[1], refXYZ[2])
  
    p2 = [0 for i in range(LED_COUNT)]
    for i in range(LED_COUNT):
        p2[i] = float(p[i])

    sr2 = illuminant.SR2(refIntensity,resIntensity)
    sr2l = illuminant.SR2L(refIntensity,resIntensity)
    return {"result":{"p":p2, "sr2":sr2, "sr2l":sr2l},"out":{"intensity":resIntensity, "p":p2, "luminance":resluminance, "XYZ":resXYZ, "lxy": resxyY, "ct": resCT},"ref":{"intensity":refIntensity,"luminance":refluminance, "XYZ":refXYZ, "lxy":refxyY, "ct":refCT}}

def optimize_ct(cct, brightness, led):

    val = illuminant.blackbody(cct)
    return optimize(val, brightness, led)

def optimize_daylight(cct, brightness, led):
    val = illuminant.daylight(cct)
    return optimize(val, brightness, led)

def test():
    import sys
    import os
    import led
    led.configure()
    D65 = [49.98,52.31,54.65,68.7,82.75,87.12,91.49,92.46,93.43,90.06,86.68,95.77,104.86,110.94,117.01,117.41,117.81,116.34,114.86,115.39,115.92,112.37,108.81,109.08,109.35,108.58,107.8,106.3,104.79,106.24,107.69,106.05,104.41,104.23,104.05,102.02,100,98.17,96.33,96.06,95.79,92.24,88.69,89.35,90.01,89.8,89.6,88.65,87.7,85.49,83.29,83.49,83.7,81.86,80.03,80.12,80.21,81.25,82.28,80.28,78.28,74,69.72,70.67,71.61,72.98,74.35,67.98,61.6,65.74,69.89]
    D55 = [32.58,35.34,38.09,49.52,60.95,64.75,68.55,70.07,71.58,69.75,67.91,76.76,85.61,91.8,97.99,99.23,100.46,100.19,99.91,101.33,102.74,100.41,98.08,99.38,100.68,100.69,100.7,100.34,99.99,102.1,104.21,103.16,102.1,102.53,102.97,101.48,100,98.61,97.22,97.48,97.75,94.59,91.43,92.93,94.42,94.78,95.14,94.68,94.22,92.33,90.45,91.39,92.33,90.59,88.85,89.59,90.32,92.13,93.95,91.95,89.96,84.82,79.68,81.26,82.84,83.84,84.84,77.54,70.24,74.77,79.3]
    D50 = [24.49,27.18,29.87,39.59,49.31,52.91,56.51,58.27,60.03,58.93,57.82,66.32,74.82,81.04,87.25,88.93,90.61,90.99,91.37,93.24,95.11,93.54,91.96,93.84,95.72,96.17,96.61,96.87,97.13,99.61,102.1,101.43,100.75,101.54,102.32,101.16,100,98.87,97.74,98.33,98.92,96.21,93.5,95.59,97.69,98.48,99.27,99.16,99.04,97.38,95.72,97.29,98.86,97.26,95.67,96.93,98.19,100.6,103,101.07,99.13,93.26,87.38,89.49,91.6,92.25,92.89,84.87,76.85,81.68,86.51]
    A = [9.8,10.9,12.09,13.35,14.71,16.15,17.68,19.29,20.99,22.79,24.67,26.64,28.7,30.85,33.09,35.41,37.81,40.3,42.87,45.52,48.24,51.04,53.91,56.85,59.86,62.93,66.06,69.25,72.5,75.79,79.13,82.52,85.95,89.41,92.91,96.44,100,103.58,107.18,110.8,114.44,118.08,121.73,125.39,129.04,132.7,136.35,139.99,143.62,147.24,150.84,154.42,157.98,161.52,165.03,168.51,171.96,175.38,178.77,182.12,185.43,188.7,191.93,195.12,198.26,201.36,204.41,207.41,210.36,213.27,216.12]
    E = [1.0] * 71

    white = [0.143,0.171,0.215,0.3,0.41,0.538,0.675,0.781,0.837,0.871,0.888,0.892,0.901,0.9,0.902,0.905,0.908,0.91,0.909,0.91,0.91,0.909,0.911,0.908,0.909,0.91,0.914,0.907,0.905,0.903,0.901,0.903,0.911,0.912,0.913,0.914,0.916,0.914,0.912,0.914,0.917,0.913,0.91,0.912,0.907,0.909,0.909,0.907,0.907,0.914,0.915,0.915,0.918,0.912,0.916,0.915,0.918,0.91,0.911,0.909,0.909,0.909,0.91,0.913,0.915,0.916,0.916,0.916,0.918,0.917,0.919]
    blue = [0.116,0.141,0.172,0.209,0.241,0.267,0.29,0.304,0.314,0.322,0.331,0.336,0.341,0.343,0.341,0.33,0.317,0.298,0.279,0.26,0.239,0.218,0.196,0.175,0.156,0.14,0.127,0.115,0.103,0.091,0.08,0.072,0.065,0.06,0.055,0.052,0.049,0.046,0.044,0.043,0.043,0.043,0.043,0.043,0.044,0.044,0.045,0.046,0.047,0.049,0.052,0.056,0.061,0.065,0.071,0.076,0.081,0.084,0.086,0.084,0.08,0.078,0.076,0.076,0.079,0.084,0.09,0.097,0.105,0.115,0.126]
    green = [0.067,0.066,0.067,0.068,0.069,0.07,0.07,0.071,0.071,0.072,0.073,0.075,0.077,0.08,0.083,0.088,0.095,0.104,0.115,0.13,0.148,0.17,0.193,0.221,0.253,0.291,0.323,0.349,0.366,0.373,0.372,0.368,0.359,0.346,0.33,0.315,0.298,0.279,0.263,0.243,0.223,0.202,0.18,0.162,0.147,0.136,0.129,0.123,0.12,0.118,0.117,0.116,0.115,0.112,0.112,0.111,0.113,0.115,0.119,0.123,0.128,0.132,0.136,0.141,0.144,0.146,0.146,0.143,0.141,0.139,0.139]
    red = [0.063,0.062,0.063,0.063,0.063,0.062,0.062,0.063,0.062,0.063,0.063,0.063,0.062,0.063,0.063,0.063,0.063,0.063,0.062,0.062,0.061,0.061,0.061,0.061,0.062,0.064,0.067,0.068,0.07,0.07,0.071,0.071,0.071,0.07,0.07,0.07,0.071,0.074,0.079,0.086,0.095,0.112,0.14,0.183,0.241,0.311,0.386,0.459,0.517,0.564,0.597,0.618,0.633,0.637,0.647,0.648,0.652,0.652,0.654,0.652,0.654,0.658,0.659,0.658,0.665,0.664,0.665,0.667,0.67,0.672,0.672]
    black = [0.044,0.045,0.044,0.045,0.045,0.045,0.045,0.045,0.045,0.045,0.045,0.045,0.044,0.044,0.044,0.044,0.044,0.044,0.044,0.044,0.044,0.043,0.043,0.043,0.043,0.044,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.042,0.043,0.043,0.042,0.043,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.042,0.043,0.043,0.043]
    standardreflector = [1.0]*71

    start = time.clock()
    # res = optimizeRelative(D65, 120, led.LED)
    res = optimize(D65, 300, led.LED)
    # res = optimize_ct(5000, 100, led.LED)

    if res == None:
        print "error."
        exit(0)

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

    debug = True
    if debug:
        if os.path.exists('res') == False:
            os.mkdir('res')

        f = open('res/solver.res.txt', 'w')
        for i in range(71):
            f.write(str(intensity[i])+"\n")

        f.close()

        f = open('res/solver.ref.txt', 'w')
        for i in range(71):
            f.write(str(refIntensity[i])+"\n")

        f.close()

        f = open('res/solver.p.txt', 'w')
        for i in range(LED_COUNT):
            f.write(str(p[i])+"\n")

        f.close()

    print "factor"
    print p
    # for i in range(18):
    #     print "p%d = %f" % (i+1, p[i])

    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]
    x = xy[1]
    y = xy[2]

    print "calclation ."
    print "XYZ = %f, %f, %f" % (X, Y, Z)
    print "xy = %f, %f" % (x, y)
    print "luminance : %f cd/m2" % luminance
    print "T : %f K" % cct

    X2 = XYZ2[0]
    Y2 = XYZ2[1]
    Z2 = XYZ2[2]
    x2 = xy2[1]
    y2 = xy2[2]

    print "reference ."
    print "XYZ = %f, %f, %f" % (X2, Y2, Z2)
    print "xy = %f, %f" % (x2, y2)
    print "luminance : %f cd/m2" % luminance2
    print "T : %f K" % cct2

    print ""
    print "sr2l = %f" % (sr2l)
    if "light" in res:
        lab = res["out"]["lab"]
        lab2 = res["ref"]["lab"]
        sr2 = res["result"]["sr2"]
        sr2l = res["result"]["sr2l"]
        delta_e = res["result"]["de00"]
        print ""
        print "lab1 = %f,%f,%f" % (lab[0],lab[1],lab[2])
        print "lab2 = %f,%f,%f" % (lab2[0],lab2[1],lab2[2])
        print "delta e (CIE2000) = %f" % delta_e
        print "SR2 = %f" % sr2
        print "SR2L = %f" % sr2l

    print "process time : %f" % (time.clock() - start)

if __name__ == '__main__':
    
    test()
