import numpy
import math
import sys

color_matching_function_X2 = numpy.array([
0.001368,
0.002236,
0.004243,
0.00765,
0.01431,
0.02319,
0.04351,
0.07763,
0.13438,
0.21477,
0.2839,
0.3285,
0.34828,
0.34806,
0.3362,
0.3187,
0.2908,
0.2511,
0.19536,
0.1421,
0.09564,
0.05795001,
0.03201,
0.0147,
0.0049,
0.0024,
0.0093,
0.0291,
0.06327,
0.1096,
0.1655,
0.2257499,
0.2904,
0.3597,
0.4334499,
0.5120501,
0.5945,
0.6784,
0.7621,
0.8425,
0.9163,
0.9786,
1.0263,
1.0567,
1.0622,
1.0456,
1.0026,
0.9384,
0.8544499,
0.7514,
0.6424,
0.5419,
0.4479,
0.3608,
0.2835,
0.2187,
0.1649,
0.1212,
0.0874,
0.0636,
0.04677,
0.0329,
0.0227,
0.01584,
0.01135916,
0.008110916,
0.005790346,
0.004106457,
0.002899327,
0.00204919,
0.001439971])
color_matching_function_Y2 = numpy.array([
0.000039,
0.000064,
0.00012,
0.000217,
0.000396,
0.00064,
0.00121,
0.00218,
0.004,
0.0073,
0.0116,
0.01684,
0.023,
0.0298,
0.038,
0.048,
0.06,
0.0739,
0.09098,
0.1126,
0.13902,
0.1693,
0.20802,
0.2586,
0.323,
0.4073,
0.503,
0.6082,
0.71,
0.7932,
0.862,
0.9148501,
0.954,
0.9803,
0.9949501,
1,
0.995,
0.9786,
0.952,
0.9154,
0.87,
0.8163,
0.757,
0.6949,
0.631,
0.5668,
0.503,
0.4412,
0.381,
0.321,
0.265,
0.217,
0.175,
0.1382,
0.107,
0.0816,
0.061,
0.04458,
0.032,
0.0232,
0.017,
0.01192,
0.00821,
0.005723,
0.004102,
0.002929,
0.002091,
0.001484,
0.001047,
0.00074,
0.00052
])
color_matching_function_Z2 = numpy.array([
0.006450001,
0.01054999,
0.02005001,
0.03621,
0.06785001,
0.1102,
0.2074,
0.3713,
0.6456,
1.0390501,
1.3856,
1.62296,
1.74706,
1.7826,
1.77211,
1.7441,
1.6692,
1.5281,
1.28764,
1.0419,
0.8129501,
0.6162,
0.46518,
0.3533,
0.272,
0.2123,
0.1582,
0.1117,
0.07824999,
0.05725001,
0.04216,
0.02984,
0.0203,
0.0134,
0.008749999,
0.005749999,
0.0039,
0.002749999,
0.0021,
0.0018,
0.001650001,
0.0014,
0.0011,
0.001,
0.0008,
0.0006,
0.00034,
0.00024,
0.00019,
0.0001,
5E-05,
0.00003,
0.00002,
0.00001,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0
])

color_matching_function_X10 = numpy.array([
0.000159952,
0.00066244,
0.0023616,
0.0072423,
0.0191097,
0.0434,
0.084736,
0.140638,
0.204492,
0.264737,
0.314679,
0.357719,
0.383734,
0.386726,
0.370702,
0.342957,
0.302273,
0.254085,
0.195618,
0.132349,
0.080507,
0.041072,
0.016172,
0.005132,
0.003816,
0.015444,
0.037465,
0.071358,
0.117749,
0.172953,
0.236491,
0.304213,
0.376772,
0.451584,
0.529826,
0.616053,
0.705224,
0.793832,
0.878655,
0.951162,
1.01416,
1.0743,
1.11852,
1.1343,
1.12399,
1.0891,
1.03048,
0.95074,
0.856297,
0.75493,
0.647467,
0.53511,
0.431567,
0.34369,
0.268329,
0.2043,
0.152568,
0.11221,
0.0812606,
0.05793,
0.0408508,
0.028623,
0.0199413,
0.013842,
0.00957688,
0.0066052,
0.00455263,
0.0031447,
0.00217496,
0.0015057,
0.00104476
])
color_matching_function_Y10 = numpy.array([
0.000017364,
0.00007156,
0.0002534,
0.0007685,
0.0020044,
0.004509,
0.008756,
0.014456,
0.021391,
0.029497,
0.038676,
0.049602,
0.062077,
0.074704,
0.089456,
0.106256,
0.128201,
0.152761,
0.18519,
0.21994,
0.253589,
0.297665,
0.339133,
0.395379,
0.460777,
0.53136,
0.606741,
0.68566,
0.761757,
0.82333,
0.875211,
0.92381,
0.961988,
0.9822,
0.991761,
0.99911,
0.99734,
0.98238,
0.955552,
0.915175,
0.868934,
0.825623,
0.777405,
0.720353,
0.658341,
0.593878,
0.527963,
0.461834,
0.398057,
0.339554,
0.283493,
0.228254,
0.179828,
0.140211,
0.107633,
0.081187,
0.060281,
0.044096,
0.0318004,
0.0226017,
0.0159051,
0.0111303,
0.0077488,
0.0053751,
0.00371774,
0.00256456,
0.00176847,
0.00122239,
0.00084619,
0.00058644,
0.00040741
])
color_matching_function_Z10 = numpy.array([
0.000704776,
0.0029278,
0.0104822,
0.032344,
0.0860109,
0.19712,
0.389366,
0.65676,
0.972542,
1.2825,
1.55348,
1.7985,
1.96728,
2.0273,
1.9948,
1.9007,
1.74537,
1.5549,
1.31756,
1.0302,
0.772125,
0.5706,
0.415254,
0.302356,
0.218502,
0.159249,
0.112044,
0.082248,
0.060709,
0.04305,
0.030451,
0.020584,
0.013676,
0.007918,
0.003988,
0.001091,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0
])

color_matching_function_X = color_matching_function_X2
color_matching_function_Y = color_matching_function_Y2
color_matching_function_Z = color_matching_function_Z2

WhitePoint = {"X":300,"Y":300,"Z":300}

def spectrum2XYZ(value):
    
    value = numpy.array(value)

    X = numpy.sum(value * color_matching_function_X) * 5 * 683
    Y = numpy.sum(value * color_matching_function_Y) * 5 * 683
    Z = numpy.sum(value * color_matching_function_Z) * 5 * 683

    return [X, Y, Z]

def spectrum2XYZ10(value):
    
    value = numpy.array(value)

    X = numpy.sum(value * color_matching_function_X10) * 5 * 683
    Y = numpy.sum(value * color_matching_function_Y10) * 5 * 683
    Z = numpy.sum(value * color_matching_function_Z10) * 5 * 683

    return [X, Y, Z]

def XYZ2lxy(X,Y,Z):
    # X = XYZ[0]
    # Y = XYZ[1]
    # Z = XYZ[2]

    if X + Y + Z == 0:
        return [0,0,0]

    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)

    return [Y, x, y]

def XYZ2LUV(X,Y,Z):
    l = Y
    u = 4 * X / (X + 15 * Y + 3 * Z)
    v = 9 * Y / (X + 15 * Y + 3 * Z)

    return [l, u, v]
    
def XYZ2LAB(XYZ, XYZ0=[500.0,500.0,500.0]):
    X0 = XYZ0[0]
    Y0 = XYZ0[1]
    Z0 = XYZ0[2]

    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]

    if X+Y+Z == 0:
        return [0, 0, 0]

    fx = 0.0
    if X / X0 > 0.008856:
        fx = pow(X / X0, 1.0 / 3.0)
    else:
        fx = (7.787 * (X / X0) + 16.0) / 116.0

    fy = 0.0
    if Y / Y0 > 0.008856:
        fy = pow(Y / Y0, 1.0 / 3.0)
    else:
        fy = (7.787 * (Y / Y0) + 16.0) / 116.0

    fz = 0.0
    if Z / Z0 > 0.008856:
        fz = pow(Z / Z0, 1.0 / 3.0)
    else:
        fz = (7.787 * (Z / Z0) + 16.0) / 116.0

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return [L, a, b]

def Luminance(values):
    # Y = 0.0
    # for i in range(71):
    #     Y = Y + values[i] * color_matching_function_Y[i]
    
    # Y = Y * 683 * 5

    values = numpy.array(values)

    Y = numpy.sum(values * color_matching_function_Y) * 683 * 5

    return Y

def Radiance(values):
    # res = 0.0
    # for i in range(71):
    #     res = res + values[i]

    values = numpy.array(values)

    res = numpy.sum(values)
    
    return res

# def cct(x, y):
#     # Hernandez-Andres et al. (1999) method

#     n = (x - 0.3366) / (y - 0.1735)
#     CCT = (-949.86315 + 6253.80338 * numpy.exp(-n / 0.92159) + 28.70599 * numpy.exp(-n / 0.20039) + 0.00004 * numpy.exp(-n / 0.07125))
#     n = numpy.where(CCT > 50000, (x - 0.3356) / (y - 0.1691), n)
#     CCT = numpy.where(CCT > 50000, 36284.48953 + 0.00228 * numpy.exp(-n / 0.07861) + 5.4535e-36 * numpy.exp(-n / 0.01543), CCT)

#     res = float(CCT)

#     if res > 25000:
#         res = 0

#     if res < 0:
#         res = 0
        
#     return res

def __lerp(a,b,c):
    return (((b) - (a)) * (c) + (a))

def cct(X,Y,Z):
    rt = [sys.float_info.min,  10.0e-6,  20.0e-6,  30.0e-6,  40.0e-6,  50.0e-6,
         60.0e-6,  70.0e-6,  80.0e-6,  90.0e-6, 100.0e-6, 125.0e-6,
        150.0e-6, 175.0e-6, 200.0e-6, 225.0e-6, 250.0e-6, 275.0e-6,
        300.0e-6, 325.0e-6, 350.0e-6, 375.0e-6, 400.0e-6, 425.0e-6,
        450.0e-6, 475.0e-6, 500.0e-6, 525.0e-6, 550.0e-6, 575.0e-6,
        600.0e-6]

    uvt = [
         [0.18006, 0.26352, -0.24341],
        [0.18066, 0.26589, -0.25479],
        [0.18133, 0.26846, -0.26876],
        [0.18208, 0.27119, -0.28539],
        [0.18293, 0.27407, -0.30470],
        [0.18388, 0.27709, -0.32675],
        [0.18494, 0.28021, -0.35156],
        [0.18611, 0.28342, -0.37915],
        [0.18740, 0.28668, -0.40955],
        [0.18880, 0.28997, -0.44278],
        [0.19032, 0.29326, -0.47888],
        [0.19462, 0.30141, -0.58204],
        [0.19962, 0.30921, -0.70471],
        [0.20525, 0.31647, -0.84901],
        [0.21142, 0.32312, -1.0182],
        [0.21807, 0.32909, -1.2168],
        [0.22511, 0.33439, -1.4512],
        [0.23247, 0.33904, -1.7298],
        [0.24010, 0.34308, -2.0637],
        [0.24792, 0.34655, -2.4681],
        [0.25591, 0.34951, -2.9641],
        [0.26400, 0.35200, -3.5814],
        [0.27218, 0.35407, -4.3633],
        [0.28039, 0.35577, -5.3762],
        [0.28863, 0.35714, -6.7262],
        [0.29685, 0.35823, -8.5955],
        [0.30505, 0.35907, -11.324],
        [0.31320, 0.35968, -15.628],
        [0.32129, 0.36011, -23.325],
        [0.32931, 0.36038, -40.770],
        [0.33724, 0.36051, -116.45]
    ]

    # if ((X < 1.0e-20) & (Y < 1.0e-20) & (Z < 1.0e-20)):
    #     return None

    us = (4.0 * X) / (X + 15.0 * Y + 3.0 * Z)
    vs = (6.0 * Y) / (X + 15.0 * Y + 3.0 * Z)
    dm = 0.0
    for i in range(31):
        di = (vs - uvt[i][1]) - uvt[i][2] * (us - uvt[i][0])
        if ((i > 0) & (((di < 0.0) & (dm >= 0.0)) | ((di >= 0.0) & (dm < 0.0)))):
            break
        dm = di
    if i == 30:
        return 0.0

    di = di / math.sqrt(1.0 + uvt[i][2] * uvt[i][2])
    dm = dm / math.sqrt(1.0 + uvt[i-1][2] * uvt[i-1][2])
    p = dm / (dm - di)
    p = 1.0 / __lerp(rt[i-1],rt[i],p)

    return p

def LAB2XYZ(lab, whitePointXYZ=[100,100,100]):
    l = lab[0]
    a = lab[1]
    b = lab[2]

    fy = (l + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    Xn = whitePointXYZ[0]
    Yn = whitePointXYZ[1]
    Zn = whitePointXYZ[2]

    Y = 0.0
    if fy > 6.0 / 29.0:
        Y = pow(fy, 3.0) * Yn
    else:
        Y = pow(3.0 / 29.0, 3.0)
    
    X = 0.0
    if fx > 6.0 / 29.0:
        X = pow(fx, 3.0) * Xn
    else:
        X = pow(3.0 / 29.0, 3.0)
        
    Z = 0.0
    if fz > 6.0 / 29.0:
        Z = pow(fz, 3.0) * Zn
    else:
        Z = pow(3.0 / 29.0, 3.0)
        

    return [X,Y,Z]

def delta_e_cie2000(lab11, lab12, Kl=1, Kc=1, Kh=1):

    lab1 = numpy.array([(lab11[0], lab11[1], lab11[2])])
    lab2 = numpy.array([(lab12[0], lab12[1], lab12[2])])

    L, a, b = lab11

    avg_Lp = (L + lab2[:, 0]) / 2.0

    C1 = numpy.sqrt(numpy.sum(numpy.power(lab1[1:], 2)))
    C2 = numpy.sqrt(numpy.sum(numpy.power(lab2[:, 1:], 2), axis=1))

    avg_C1_C2 = (C1 + C2) / 2.0

    G = 0.5 * (1 - numpy.sqrt(numpy.power(avg_C1_C2, 7.0) / (numpy.power(avg_C1_C2, 7.0) + numpy.power(25.0, 7.0))))

    a1p = (1.0 + G) * a
    a2p = (1.0 + G) * lab2[:, 1]

    C1p = numpy.sqrt(numpy.power(a1p, 2) + numpy.power(b, 2))
    C2p = numpy.sqrt(numpy.power(a2p, 2) + numpy.power(lab2[:, 2], 2))

    avg_C1p_C2p = (C1p + C2p) / 2.0

    h1p = numpy.degrees(numpy.arctan2(b, a1p))
    h1p += (h1p < 0) * 360

    h2p = numpy.degrees(numpy.arctan2(lab2[:, 2], a2p))
    h2p += (h2p < 0) * 360

    avg_Hp = (((numpy.fabs(h1p - h2p) > 180) * 360) + h1p + h2p) / 2.0

    T = 1 - 0.17 * numpy.cos(numpy.radians(avg_Hp - 30)) + \
        0.24 * numpy.cos(numpy.radians(2 * avg_Hp)) + \
        0.32 * numpy.cos(numpy.radians(3 * avg_Hp + 6)) - \
        0.2 * numpy.cos(numpy.radians(4 * avg_Hp - 63))

    diff_h2p_h1p = h2p - h1p
    delta_hp = diff_h2p_h1p + (numpy.fabs(diff_h2p_h1p) > 180) * 360
    delta_hp -= (h2p > h1p) * 720

    delta_Lp = lab2[:, 0] - L
    delta_Cp = C2p - C1p
    delta_Hp = 2 * numpy.sqrt(C2p * C1p) * numpy.sin(numpy.radians(delta_hp) / 2.0)

    S_L = 1 + ((0.015 * numpy.power(avg_Lp - 50, 2)) / numpy.sqrt(20 + numpy.power(avg_Lp - 50, 2.0)))
    S_C = 1 + 0.045 * avg_C1p_C2p
    S_H = 1 + 0.015 * avg_C1p_C2p * T

    delta_ro = 30 * numpy.exp(-(numpy.power(((avg_Hp - 275) / 25), 2.0)))
    R_C = numpy.sqrt((numpy.power(avg_C1p_C2p, 7.0)) / (numpy.power(avg_C1p_C2p, 7.0) + numpy.power(25.0, 7.0)))
    R_T = -2 * R_C * numpy.sin(2 * numpy.radians(delta_ro))

    return numpy.sqrt(
        numpy.power(delta_Lp / (S_L * Kl), 2) +
        numpy.power(delta_Cp / (S_C * Kc), 2) +
        numpy.power(delta_Hp / (S_H * Kh), 2) +
        R_T * (delta_Cp / (S_C * Kc)) * (delta_Hp / (S_H * Kh)))

def xyY2XYZ(x,y,Y):

    X = (x / y) * Y
    Z = (1 - x - y) / y * Y
    
    XYZ = [X,Y,Z]
    # m = max(XYZ)

    # XYZ[0] = XYZ[0] / m * Y
    # XYZ[1] = XYZ[1] / m * Y
    # XYZ[2] = XYZ[2] / m * Y

    # print XYZ
    return XYZ

def RGB2XYZ(R,G,B):
    K = 1.00#5.6508#1.0/0.17697
    M = [[0.490,0.310,0.200],[0.17697,0.8124,0.01063],[0.000,0.0100,0.990]]

    X = K * (0.490 * R + 0.310 * G + 0.200 * B)
    Y = K * (0.17697 * R + 0.8124 * G + 0.01063 * B)
    Z = K * (0.000 * R + 0.010 * G + 0.990 * B)

    # X = 0.412391 * R + 0.357584 * G + 0.180481 * B
    # Y = 0.212639 * R + 0.715169 * G + 0.072192 * B
    # Z = 0.019331 * R + 0.119195 * G + 0.950532 * B
    
    XYZ = [X,Y,Z]
    m = max(XYZ)

    XYZ[0] = XYZ[0] / m * Y
    XYZ[1] = XYZ[1] / m * Y
    XYZ[2] = XYZ[2] / m * Y

    return XYZ

def SR2(val1, val2, val3=[1.0]*71):

    res = 0.0
    n=71
    Yv=1.0
    a=0.0
    b=0.0
    v=sum(color_matching_function_Y)
    for i in range(n):
        a = a + color_matching_function_Y[i] / v * val3[i]
        b = b + val3[i] / float(n)
    Yv = a / b
    for i in range(n):
        res = res + pow((val1[i] - val2[i])/Yv, 2)

    res = res / float(n)

    return res

def SR2L(val1, val2):

    res = 0.0
    n = len(val1)
    Yn = sum(val1) / float(n)
    Yv = 0.0
    # Yv = 1.0
    v=sum(color_matching_function_Y)
    for i in range(n):
        Yv = Yv + color_matching_function_Y[i] / v * val1[i]
    Yv = Yv / Yn

    for i in range(n):
        res = res + pow((val1[i] - val2[i])/(Yn*Yv),2)

    res = res / float(n)

    return res

def blackbody(Tc):
    C1 = 3.7E-16
    C2 = 0.0144
    wl = 380.0
    val = [0.0 for i in range(71)]
    index560 = 0
    for i in range(71):
        wl2 = wl / 1000000000.0
        val[i] = (C1 / (math.pi * wl2 * wl2 * wl2 * wl2 * wl2) * 1.0 / (math.exp(C2 / (wl2 * float(Tc))) - 1.0))
        if wl == 560:
            index560 = i
        wl += 5

    n = 0.0
    n = val[index560]
    for i in range(71):
        val[i] = val[i] / n * 100.0

    return val
    
def daylight(Tc):
    Tc = float(Tc)

    S0 = [
        63.4,
64.6,
65.8,
80.3,
94.8,
99.8,
104.8,
105.35,
105.9,
101.35,
96.8,
105.35,
113.9,
119.75,
125.6,
125.55,
125.5,
123.4,
121.3,
121.3,
121.3,
117.4,
113.5,
113.3,
113.1,
111.95,
110.8,
108.65,
106.5,
107.65,
108.8,
107.05,
105.3,
104.85,
104.4,
102.2,
100,
98,
96,
95.55,
95.1,
92.1,
89.1,
89.8,
90.5,
90.4,
90.3,
89.35,
88.4,
86.2,
84,
84.55,
85.1,
83.5,
81.9,
82.25,
82.6,
83.75,
84.9,
83.1,
81.3,
76.6,
71.9,
73.1,
74.3,
75.35,
76.4,
69.85,
63.3,
67.5,
71.7]
    S1 = [
        38.5,
36.75,
35,
39.2,
43.4,
44.85,
46.3,
45.1,
43.9,
40.5,
37.1,
36.9,
36.7,
36.3,
35.9,
34.25,
32.6,
30.25,
27.9,
26.1,
24.3,
22.2,
20.1,
18.15,
16.2,
14.7,
13.2,
10.9,
8.6,
7.35,
6.1,
5.15,
4.2,
3.05,
1.9,
0.95,
0,
-0.8,
-1.6,
-2.55,
-3.5,
-3.5,
-3.5,
-4.65,
-5.8,
-6.5,
-7.2,
-7.9,
-8.6,
-9.05,
-9.5,
-10.2,
-10.9,
-10.8,
-10.7,
-11.35,
-12,
-13,
-14,
-13.8,
-13.6,
-12.8,
-12,
-12.65,
-13.3,
-13.1,
-12.9,
-11.75,
-10.6,
-11.1,
-11.6]
    S2 = [
        3,
2.1,
1.2,
0.05,
-1.1,
-0.8,
-0.5,
-0.6,
-0.7,
-0.95,
-1.2,
-1.9,
-2.6,
-2.75,
-2.9,
-2.85,
-2.8,
-2.7,
-2.6,
-2.6,
-2.6,
-2.2,
-1.8,
-1.65,
-1.5,
-1.4,
-1.3,
-1.25,
-1.2,
-1.1,
-1,
-0.75,
-0.5,
-0.4,
-0.3,
-0.15,
0,
0.1,
0.2,
0.35,
0.5,
1.3,
2.1,
2.65,
3.2,
3.65,
4.1,
4.4,
4.7,
4.9,
5.1,
5.9,
6.7,
7,
7.3,
7.95,
8.6,
9.2,
9.8,
10,
10.2,
9.25,
8.3,
8.95,
9.6,
9.05,
8.5,
7.75,
7,
7.3,
7.6]

    xD = 0.0
    yD = 0.0

    if Tc >= 4000 and Tc <= 7000:
        xD = -4.6070 * 10.0 ** 9.0 / Tc ** 3.0 + 2.9678 * 10.0 ** 6.0 / Tc ** 2.0 + 0.09911 * 10.0 ** 3.0 / Tc + 0.244063
    
    elif Tc > 7000:
        xD = -2.0064 * 10.0 ** 9.0 / Tc ** 3.0 + 1.9018 * 10.0 ** 6.0 / Tc ** 2.0 + 0.24748 * 10.0 ** 3.0 / Tc + 0.23704

    elif Tc < 4000:
        return blackbody(Tc)

    else:
        raise ValueError
    
    yD = -3.000 * xD * xD + 2.870 * xD - 0.275

    M1 = (-1.3515 - 1.7703 * xD + 5.9114 * yD) / (0.0241 + 0.2562 * xD  - 0.7341 * yD)
    M2 = (0.0300 - 31.4424 * xD + 30.0717 * yD) / (0.0241 + 0.2562 * xD  - 0.7341 * yD)

    val = [0.0 for i in range(71)]

    for i in range(71):
        val[i] = float(S0[i]) + M1 * float(S1[i]) + M2 * float(S2[i])
        
    # print xD
    # print yD
    # print M1
    # print M2

    return val

if __name__ == "__main__":
    sp = [0 for i in range(71)]

    f = open("test.txt", "r")

    for i in range(71):
        sp[i] = float(f.readline())

    f.close()

    X,Y,Z = spectrum2XYZ(sp)
    print("XYZ = %f, %f, %f" % (X, Y, Z))

    l,x, y = XYZ2lxy(X,Y,Z)
    print("xyl = %f, %f, %f" % (x, y, l))

    # l,a,b = XYZ2LAB(X,Y,Z)
    # print "L*a*b* = %f, %f, %f" % (l, a, b)

    luminance = Luminance(sp)
    print("luminance = %f cd/m2" % (luminance))
    
    Radiance = Radiance(sp)
    print("Radiance = %f W/sr/m2" % (Radiance))

    t = cct(X,Y,Z)
    print("T = %f K" % (t))

    print(blackbody(6500))

    print(daylight(6500))