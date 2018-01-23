import numpy as np
import cupy as cp
xp = np

from PIL import Image
import math
import os

class EditImg:
    charSize = 0
    moveLength = 0
    #色相分の水増し数(1,2,3,6,12)
    hs = 3
    #背景色(黒)
    back =(0,0,0)
    def __init__(self,charSize,moveLength,gpu):
        self.charSize = round(charSize)
        self.moveLength = round(moveLength)
        if(gpu>=0):
            xp = cp

    def expCanvas(self,inImg):
        xChar = math.ceil(inImg.size[0]/self.charSize)
        yChar = math.ceil(inImg.size[1]/self.charSize)

        xFix = xChar*self.charSize
        yFix = yChar*self.charSize

        xSt = math.floor((xFix-inImg.size[0])/2)
        ySt = math.floor((yFix-inImg.size[1])/2)

        outImg = Image.new('HSV',(xFix,yFix),self.back)
        outImg.paste(inImg,(xSt,ySt))

        return outImg

    def expCanvas2(self,inImg):
        xFix = inImg.size[0]+self.charSize
        yFix = inImg.size[1]+self.charSize

        outImg = Image.new('HSV',(xFix,yFix),self.back)
        outImg.paste(inImg,(0,0))

        return outImg

    def writeImg2(self,inImg,outDel):
        if(os.path.isdir(outDel)==False):
            os.mkdir(outDel)
        outImgs =[[[0 for i in range(8)]for k in range(2)] for j in range(self.hs)]
        outImgs[0][0][0] = inImg
        outImgs[0][0][4] = outImgs[0][0][0].transpose(Image.ROTATE_90)
        for i in range(0, 8, 4):
            outImgs[0][0][i+1] = outImgs[0][0][i].transpose(Image.FLIP_LEFT_RIGHT)
            outImgs[0][0][i+2] = outImgs[0][0][i].transpose(Image.FLIP_TOP_BOTTOM)
            outImgs[0][0][i+3] = outImgs[0][0][i+1].transpose(Image.FLIP_TOP_BOTTOM)
        ctr = 0
        files = os.listdir(outDel)
        for i in files:
            ctr = ctr + 1
        for j in range(0,self.hs):
            for k in range(0,2):
                for i in range(0,8):
                    h, s, v = outImgs[0][0][i].split()
                    j2 = (255/self.hs)*j
                    h2 = h.point(lambda h: round((h+j2)%255,0))
                    s2 = s.point(lambda s: round((s*(k+1))%255,0))
                    outImgs[j][k][i] = Image.merge("HSV", (h2 , s2, v))
                    outImg = outImgs[j][k][i].convert("RGB")
                    outImg.save(outDel+'/'+str(ctr)+'.png')
                    ctr = ctr + 1

    def quarryImgCNN(self,inImg):
        xCh = int((inImg.size[0]-self.charSize)/self.moveLength+1)
        yCh = int((inImg.size[1]-self.charSize)/self.moveLength+1)
        outImgs = [[0 for i in range(xCh)] for j in range(yCh)]
        for j in range(0,yCh):
                for i in range(0,xCh):
                    xSt = i*self.moveLength
                    ySt = j*self.moveLength
                    outImgs[j][i] = inImg.crop((xSt, ySt,
                                                (xSt+self.charSize), (ySt+self.charSize)))
        return outImgs

    def quarryImg2(self,inImg):
        outImgs = []
        for j in range(0,inImg.size[0]-self.charSize,self.moveLength):
                for i in range(0,inImg.size[1]-self.charSize,self.moveLength):
                    outImgs.append(inImg.crop((i, j, (i+self.charSize), (j+self.charSize))))
        return outImgs

    def pile2img(self,numpy):
        ySize = numpy.shape[1]-self.charSize*2+2
        xSize = numpy.shape[2]-self.charSize*2+2
        outImg = Image.new('HSV',(xSize,ySize),self.back)
        for y in range(ySize):
            for x in range(xSize):
                outImg.putpixel((x, y),(
                    int(round(numpy[0][y+self.charSize-1][x+self.charSize-1]*255)),
                    int(round(numpy[1][y+self.charSize-1][x+self.charSize-1]*255)),
                    int(round(numpy[2][y+self.charSize-1][x+self.charSize-1]*255)),
                    ))
        return outImg

    def addCanvas(self,inImg):
        if(inImg.size[0] >= self.charSize):
            xSize = inImg.size[0]+(self.charSize-1)*2
            xSt = inImg.size[0]-1
        else:
            xSize = self.charSize+(self.charSize-1)*2
            xSt = self.charSize-1
        if(inImg.size[1] >= self.charSize):
            ySize = inImg.size[1]+(self.charSize-1)*2
            ySt = inImg.size[1]-1
        else:
            ySize = self.charSize+(self.charSize-1)*2
            ySt = self.charSize-1
        outImg = Image.new('HSV',(xSize,ySize),self.back)
        outImg.paste(inImg,(xSt,ySt))
        return outImg

    def img2cnn(self,Img):
        h,s,v = Img.split()
        h2 = xp.asarray(xp.float32(h)/255.0)
        s2 = xp.asarray(xp.float32(s)/255.0)
        v2 = xp.asarray(xp.float32(v)/255.0)
        cnn = xp.asarray([h2, s2, v2],dtype=xp.float32)
        return cnn


    def outImg2(self,outDel,path):
        img = Image.open(path).convert("HSV")
        img = self.expCanvas2(img)
        imgsQua = self.quarryImg2(img)
        for x in range(0, len(imgsQua)):
            fileNames = os.path.basename(path)
            fileNames = os.path.splitext(fileNames)
            self.writeImg2(imgsQua[x],outDel)

    def preTeachImgCNN(self,path):
        img = Image.open(path).convert("HSV")
        img = self.expCanvas(img)
        img = self.addCanvas(img)
        imgsQua = self.quarryImgCNN(img)
        return imgsQua

"""
edit = EditImg(32)
img = Image.open('alldata/0000_i.bmp').convert("HSV")
img = edit.expCanvas(img)
imgsqua = edit.quarryImg(img,32)
imgOut = edit.sutureImg(imgsqua)
imgOut = imgOut.convert("RGB")
imgOut.save('alldata/sample_out.png')
"""


"""
edit = EditImg(32)
inImg = Image.open('alldata/sample.png').convert("HSV")
numpy = edit.img2numpy(inImg)
image = edit.numpy2img(numpy)
image = image.convert("RGB")
image.save('alldata/sample_out.png')
print('書き出し完了')
"""