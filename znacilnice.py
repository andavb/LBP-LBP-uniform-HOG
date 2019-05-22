from cv2 import *
import numpy as np
from PIL import Image
import numpy
from matplotlib import pyplot as plt
from resizeimage import resizeimage

utezi = [1, 2, 4, 8, 16, 32, 64, 128]#za bite npr 00000101 pomnozis z 1 in z 4
uniformni = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255]
UNI = []
histogramuniformni = []
LBPuniform = []
q=False
print(len(uniformni))

for a in range(0, 256): # priredis vrednosti 59 v UNI polje
    for x in range(0, len(uniformni)):
        if uniformni[x] == a:
            UNI.append(a)
            q = True
            break

    if q != True:
        UNI.append(59)
    else:
        q = False

print(UNI)

img = cv2.imread("na.jpg")

grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
LBPosnovna = grayImg.copy()
LBPusmerjenih = grayImg.copy()


height, width, channels = img.shape


def izracunajLBP():
    global grayImg, LBPosnovna

    for y in range(1, height):
        for x in range(1, width):
            if(y < height-1 and x < width-1):

                sredina = grayImg[y, x]
                zgorajLevo = grayImg[y-1, x-1]
                zgoraj = grayImg[y-1, x]
                zgorajDesno = grayImg[y-1, x+1]
                levo= grayImg[y, x-1]
                desno = grayImg[y, x+1]
                spodajLevo = grayImg[y+1, x-1]
                spodaj= grayImg[y+1, x]
                spodajDesno= grayImg[y+1, x+1]

                byte = []

                for i in [zgorajLevo, zgoraj, zgorajDesno, levo, desno, spodajLevo, spodaj, spodajDesno]:
                    if i >= sredina:
                        byte.append(1)
                    else:
                        byte.append(0)

                rezultat = 0

                for i in range(0, 8):
                    rezultat += utezi[i]*byte[i]

                LBPosnovna[y, x] = rezultat


def izracunajLBPuniform():
    global LBPosnovna, grayImg, UNI, histogramuniformni, LBPuniform
    Hst = [0] * 256

    LBPuniform = LBPosnovna.copy()

    for y in range(0, height):
        for x in range(0, width):
                Hst[UNI[LBPosnovna[y, x]]]+=1
                LBPuniform[y,x] = UNI[LBPosnovna[y,x]]

    for x in range(0, len(Hst)):
        if Hst[x] != 0:
            histogramuniformni.append(Hst[x])

def izracunajLBPusmerjenihGrad(d):
    global grayImg, LBPusmerjenih

    for y in range(1, height):
        for x in range(1, width):
            if (y < height - 1 and x < width - 1):

                zgorajLevo = grayImg[y - 1, x - 1]
                zgoraj = grayImg[y - 1, x]
                zgorajDesno = grayImg[y - 1, x + 1]
                levo = grayImg[y, x - 1]
                desno = grayImg[y, x + 1]
                spodajLevo = grayImg[y + 1, x - 1]
                spodaj = grayImg[y + 1, x]
                spodajDesno = grayImg[y + 1, x + 1]

                byte = []
                polja = [zgorajLevo, zgoraj, zgorajDesno, levo, desno, spodajLevo, spodaj, spodajDesno]

                stevec = 0

                for i in polja:
                    primerjaj = stevec+d

                    #print(primerjaj)
                    if primerjaj > 7:
                        primerjaj = (primerjaj - 7) - 1
                        #print("ta je index potem:")
                        #print(primerjaj)

                    if i >= polja[primerjaj]:
                        byte.append(1)
                    else:
                        byte.append(0)
                    stevec += 1

                rezultat = 0

                for i in range(0, 8):
                    rezultat += utezi[i] * byte[i]

                LBPusmerjenih[y, x] = rezultat


def pridobiKote(x, y):
    if x == 0:
        x=1
    kot = np.round(np.arctan2(y, x) * (180/np.pi))

    return kot

pridobiKote = np.vectorize(pridobiKote)


def gradientX(G, h, w):
    Gradient = G.copy()

    for x in range(h):
        for y in range(w):
            if (x == h-1 or y == w-1):
                Gradient[x][y] = 0
            else:
                Gradient[x][y] = abs(int(G[x][y -1]*(-1)) + int(G[x][y+1])) #[-1,0,1] pomnozis pa sestejes

    return Gradient

def gradientY(G, h, w):
    Gradient = G.copy()

    for x in range(h):
        for y in range(w):
            if (x == h-1 or y == w-1):
                Gradient[x][y] = 0
            else:
                Gradient[x][y] = abs(int(G[x-1][y] * (-1)) + int(G[x+1][y]))  # [[0,-1,0],[0,0,0],[0,1,0]] pomnozis pa sestejes

    return Gradient

def HOG(bins, cellsize, regionsize):
    global grayImg
    widthR = width
    heightR = height

    while(True): #izracunam da je sirina slike deljiva z cellsize
        if widthR % cellsize == 0:
            break;
        else:
            widthR+=1

    while (True):
        if heightR % cellsize == 0:
            break;
        else:
            heightR += 1

    print(heightR,widthR)


    imgResized = cv2.resize(grayImg, (widthR, heightR))

    Gx = np.asarray(imgResized).copy()
    Gy = np.asarray(imgResized).copy()

    Gx = gradientX(Gx, heightR, widthR)
    Gy = gradientY(Gy, heightR, widthR)

    Gradient = np.round(np.sqrt(np.multiply(Gx, Gx) + np.multiply(Gy, Gy)))

    A = pridobiKote(Gx, Gy)

    razmakBina = 180/bins

    '''for Y in range(heightR):
        for X in range(widthR):
            if A[Y][X] > 80:
                print(A[Y][X])
    '''
    stevec = 0
    koncneVrenosti = []


    for regijaY in range(0, heightR-cellsize, cellsize):
        for regijaX in range(0, widthR-cellsize, cellsize):
            stevec+=1
            Hist = [[0.0] * bins for i in range(regionsize * regionsize)]
            index1Hisotgrama = 0
            vsotaVrednosti = 0

            zactekRegijeY = regijaY
            zactekRegijeX = regijaX

            konecRegijeX = regijaX+ (cellsize * regionsize)
            konecRegijeY = regijaY+ (cellsize * regionsize)

            for celY in range(zactekRegijeY, konecRegijeY, cellsize):
                for celX in range(zactekRegijeX, konecRegijeX, cellsize):

                    zacetekCeliceY = celY
                    zacetekCeliceX = celX

                    konecCeliceY = celY + cellsize
                    konecCeliceX = celX + cellsize

                    for Y in range(zacetekCeliceY, konecCeliceY):
                        for X in range(zacetekCeliceX, konecCeliceX):
                            kot = A[Y, X]
                            gradient = Gradient[Y, X]

                            if (kot > razmakBina):
                                index = np.ceil((kot / razmakBina))
                                ostanek1 = index - (kot / razmakBina)
                                trenutni = (1 - ostanek1) * gradient
                                prejsni = ostanek1 * gradient
                            else:
                                index = np.ceil((kot / razmakBina))
                                ostanek1 = (kot / razmakBina)
                                prejsni = (1 - ostanek1) * gradient
                                trenutni = ostanek1 * gradient

                            Hist[index1Hisotgrama][int(index)] += trenutni
                            Hist[index1Hisotgrama][int(index - 1)] += prejsni

                    index1Hisotgrama += 1

            for histogram in range(0, regionsize * regionsize):
                for steviloHistggrama in range(0, bins):
                    vsotaVrednosti += np.multiply(Hist[histogram][steviloHistggrama],Hist[histogram][steviloHistggrama])

                    koncneVrenosti.append(Hist[histogram][steviloHistggrama])
            print(Hist)
            L = np.sqrt(vsotaVrednosti)

            print(L)

    print(koncneVrenosti)

    print(len(koncneVrenosti))
    print(stevec)

izracunajLBP()
izracunajLBPuniform()
izracunajLBPusmerjenihGrad(2)

HOG(9, 8, 2)

cv2.imshow("LBP osnovna", LBPosnovna)
cv2.imshow("LBP unfirom", LBPuniform)
cv2.imshow("LBP usmerjenih", LBPusmerjenih)

histogram = cv2.calcHist([LBPosnovna],[0],None,[256],[0,256])
sum = histogram.cumsum()#komulativno sesteje vse vrednosti histograma
normaliziranHistogram = (sum * histogram.max()) / sum.max()

plt.plot(normaliziranHistogram, color = 'g')
plt.hist(LBPosnovna.flatten(),256,[0,256])#speremnis 256 v 59 za uniform
plt.title("LBP osnovno")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()