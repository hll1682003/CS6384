import cv2
import numpy as np
import sys

def gamma(x): #gamma correction, process each tuple
    for i in range(0,x.shape[0]):
        if x[i] < 0.00304:
            x[i] *= 12.92
        else:
            x[i] = 1.055*pow(x[i], 1/2.4)-0.055
    return x


def invgamma(x):    #inverse gamma correction, process each tuple
    for i in range(0,x.shape[0]):
        if (x[i]<0.03928):
            x[i]/=12.92
        else:
            x[i]=pow(((x[i]+0.055)/1.055),2.4)
    return x

def XYZToLuv(x):
    if (x[0]==0 and x[1]==0 and x[2]==0):#handle the case when x=[0,0,0]
        return np.array([0,0,0],dtype=np.float_)
    if (x[1]>0.008856):
        L=116*pow(x[1],1/3)-16
    else:
        L=903.3*x[1]      
    d=x[0]+15*x[1]+3*x[2]
    uprime1=4*x[0]/d
    vprime1=9*x[1]/d
    u=13*L*(uprime1-0.19771072)
    v=13*L*(vprime1-0.46826222)
    return np.array([L,u,v],dtype=np.float_)

def LuvToXYZ(x):
    if (x[0]==0):#in case L is 0
        return np.array([0,0,0],dtype=np.float_)
    uprime2=(x[1]+13*0.19793943*x[0])/(13*x[0])  #
    vprime2=(x[2]+13*0.46831096*x[0])/(13*x[0])  #
    if (x[0]>7.9996):
        Y=pow(((x[0]+16)/116),3)
    else:
        Y=x[0]/903.3
    if (vprime2==0):
        X=0
        Z=0
    else:
        X=Y*2.25*uprime2/vprime2             
        Z=Y*(3-0.75*uprime2-5*vprime2)/vprime2
    return np.array([X,Y,Z],dtype=np.float_)

def limit(x):
    for i in range(0,x.shape[0]):
        if x[i]>1:
            x[i]=1
        else:
            if x[i]<0:
                x[i]=0
    return x

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]
rgbToXYZ=np.array([[0.412453,0.357580,0.180423],[0.212671,0.715160,0.072169],[0.019334,0.119193,0.950227]],np.float_)#convert matrix references to https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html
XYZtorgb=np.array([[3.240479,-1.53715,-0.498535],[-0.969256,1.875991,0.041556],[0.055648,-0.204043,1.057311]],np.float_)

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

rows, cols, bands = inputImage.shape # bands == 3
totalPixels=rows*cols
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels
minL=2147483647
maxL=-2147483648

tmp = np.copy(inputImage)#indicate the specified window in the original image
window=np.zeros([H2-H1+1,W2-W1+1,bands],dtype=np.float_)
for i in range(H1, H2+1) : 
    for j in range(W1, W2+1) :
        if (i==H1 or i==H2 or j==W1 or j==W2): #this outlines the specified window in the original image
            tmp[i, j] = [0,0,255]
        b, g, r = inputImage[i, j]
        window[i-H1,j-W1]=r,g,b
        window[i-H1,j-W1,0],window[i-H1,j-W1,1],window[i-H1,j-W1,2]=window[i-H1,j-W1,0]/255,window[i-H1,j-W1,1]/255,window[i-H1,j-W1,2]/255
        window[i-H1,j-W1]=np.array(limit(invgamma(window[i-H1,j-W1])),np.float_).T
        window[i-H1,j-W1]=np.array(np.dot(rgbToXYZ,window[i-H1,j-W1]))
        window[i-H1,j-W1]=np.array(XYZToLuv(window[i-H1,j-W1]).T,np.float_)
        if (window[i-H1,j-W1,0]<minL):#find out the min value of L
            minL=window[i-H1,j-W1,0]
        if (window[i-H1,j-W1,0]>maxL):#find out the max value of L
            maxL=window[i-H1,j-W1,0]
cv2.imshow('Specified Window', tmp)

print("The minimum Luminence in the window:",minL,"\nThe maximum luminence in the window: ",maxL)
Linterval=(maxL-minL)/100 #calculate the luminence interval between discretized values
accumulation=np.zeros([101],dtype=np.int_) #save the counted number of pixels for each of those values
mappingChart=np.zeros([101],dtype=np.int_)
print("Luminence interval=",Linterval)

temp3=np.zeros([inputImage.shape[0],inputImage.shape[1],inputImage.shape[2]],dtype=np.float_) #simply copy the input image but save in float format
temp4=np.zeros([inputImage.shape[0],inputImage.shape[1],inputImage.shape[2]],dtype=np.uint8) #save a stretched and ready to output matrix in integer format
for i in range(0,temp3.shape[0]):#Convert the original image to Luv and calculate the mapping chart
    for j in range(0,temp3.shape[1]):
        b,g,r=inputImage[i,j]
        temp3[i,j]=np.array([r,g,b],dtype=np.float_)
        temp3[i,j]=np.dot(1/255,temp3[i,j])
        temp3[i,j]=np.array(limit(invgamma(temp3[i,j])),dtype=np.float_)
        temp3[i,j]=np.array(np.dot(rgbToXYZ,temp3[i,j]))
        temp3[i,j]=np.array(XYZToLuv(temp3[i,j]))
        if (temp3[i,j,0]<minL):
            accumulation[0]+=1
        else:
            if (temp3[i,j,0]>maxL):
                accumulation[100]+=1
            else:
                if (minL!=maxL):#in case that all the Ls are the same(handle division by zero)
                    intervaltemp=int(round((temp3[i,j,0]-minL)/Linterval))#calculate the index to which current pixel belong
                    accumulation[intervaltemp]+=1
print("Counting:\n",accumulation)
mappingChart[0]=accumulation[0]*101/(2*totalPixels)
for i in range(1,accumulation.shape[0]):#calculate the accumulation distribution and save it
    accumulation[i]=accumulation[i-1]+accumulation[i]
    mappingChart[i]=round((accumulation[i-1]+accumulation[i])*101/(2*totalPixels)-0.5)
    if (mappingChart[i]==101):
        mappingChart[i]-=1

for i in range(0,temp3.shape[0]):
    for j in range(0,temp3.shape[1]):
        if (temp3[i,j,0]<minL):
            temp3[i,j,0]=0
        else:
            if (temp3[i,j,0]>maxL):
                temp3[i,j,0]=100
            else:
                if (minL!=maxL):
                    temp3[i,j,0]=mappingChart[int(round((temp3[i,j,0]-minL)/Linterval))]
        temp3[i,j]=np.array(LuvToXYZ(temp3[i,j]),np.float_)
        temp3[i,j]=np.array(np.dot(XYZtorgb,temp3[i,j].T),np.float_)
        temp3[i,j]=limit(temp3[i,j])
        temp3[i,j]=np.array(gamma(temp3[i,j]),np.float_)
        temp3[i,j]=np.array(np.dot(255,temp3[i,j].T),np.float_)
        temp4[i,j]=np.array([temp3[i,j,2],temp3[i,j,1],temp3[i,j,0]],np.uint8).T
cv2.imshow('Final Streching',temp4)
print("accumulation function:\n", accumulation)
print("mappingchart:\n",mappingChart)
print(minL,maxL)

# end of example of going over window

outputImage = np.zeros([rows, cols, bands], dtype=np.uint8)

for i in range(0, rows) :
    for j in range(0, cols) :
        b, g, r = inputImage[i, j]
        outputImage[i,j] = [b, g, r]
cv2.imwrite(name_output, temp4);
# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
