import cv2
import numpy as np
import sys

def gamma(x):
	if (x<0.00304):
		x*=12.92
	else:
		x=1.055*pow(x,1/2.4)-0.055
	return x

def invgamma(x):    #do the work of inverse gamma transformation
	if (x<0.03928):
		x*=1/12.92
	else:
		x=pow(((x+0.055)/1.055),2.4)
	return x

def XYZToLuv(x):
	if (x[1]>0.008856):
		L=116*pow(x[1],1/3)-16
	else:
		L=903.3*x[1]
	uprime=4.0*x[0]/(x[0]+15.0*x[1]+3.0*x[2])
	vprime=9.0*x[1]/(x[0]+15.0*x[1]+3.0*x[2])
	u=13*L*(uprime-0.19793943)
	v=13*L*(vprime-0.46831096)
	return L,u,v

def LuvToXYZ(x):
	if (x[0]==0):
		return 0,0,0
	uprime=(x[1]+13*0.19793943*x[0])/(13*x[0])  #
	vprime=(x[2]+13*0.46831096*x[0])/(13*x[0])  #
	if (x[0]>7.9996):
		Y=pow(((x[0]+16)/116),3)
	else:
		Y=x[0]/903.3
	if (vprime==0):
		X=0
		Z=0
	else:
		X=Y*2.25*uprime/vprime             #
		Z=Y*(3-0.75*uprime-5*vprime)/vprime #
	return X,Y,Z

def forwardTransform(x):
	temp=np.zeros([x.shape[0],x.shape[1],x.shape[2]],dtype=np.float_)
	minL=2147483647.0
	maxL=-2147483648.0
	for i in range(temp.shape[0]-1):
		for j in range(temp.shape[1]-1):
			temp[i,j]=XYZToLuv(np.dot(rgbToXYZ,x[i,j]/255))#each RGB tuple divided by 255, and transformed to XYZ, then to Luv
			if (temp[i,j,0]<minL): #find out the maximum and minimum of Luminance value in the specified window
				minL=temp[i,j,0]
			if (temp[i,j,0]>maxL):
				maxL=temp[i,j,0]
	print(minL,maxL)
	for i in range(temp.shape[0]-1):#do the scaling, and transform them to R8 G8 B8
		for j in range(temp.shape[1]-1):
			temp[i,j,0]=(temp[i,j,0]-minL)*100/(maxL-minL)
			temp[i,j]=LuvToXYZ(temp[i,j])
			temp[i,j]=255*np.dot(XYZtorgb,temp[i,j])

	new=np.zeros([temp.shape[0],temp.shape[1],temp.shape[2]],dtype=np.uint8)
	for i in range(0,temp.shape[0]-1):
		for j in range(0,temp.shape[1]-1):
			new[i,j]=round(temp[i,j,2]),round(temp[i,j,1]),round(temp[i,j,0])
	#new=temp.astype(np.uint8)
	return new


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
rgbToXYZ=[[0.412453,0.357580,0.180423],[0.212671,0.715160,0.072169],[0.019334,0.119193,0.950227]]#convert matrix references to https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html
XYZtorgb=[[3.240479,-1.53715,-0.498535],[-0.969256,1.875991,0.041556],[0.055648,-0.204043,1.057311]]

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

cv2.imshow("input image: " + name_input, inputImage)

rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))    #Why -1 instead of 0.5?
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

tmp = np.copy(inputImage)

XYZMatrix=np.zeros([H2-H1+1,W2-W1+1,3],dtype=np.uint8) #the windowed area of the original pic


for i in range(H1, H2) :
    for j in range(W1, W2) :
        b, g, r = inputImage[i, j]
        gray = round(0.299*r + 0.587*g + 0.114*b + 0.5)
        XYZMatrix[i-H1,j-W1]=r,g,b #adjust bgr ordering to rgb ordering
        tmp[i, j] = [gray, gray, gray]

XYZMatrix=forwardTransform(XYZMatrix)

cv2.imshow('tmp', tmp)

# end of example of going over window

#outputImage = np.zeros([rows, cols, bands], dtype=np.uint8)

#for i in range(0, rows) :
#    for j in range(0, cols) :
#        b, g, r = inputImage[i, j]
#        outputImage[i,j] = [b, g, r]
#cv2.imshow("output:", outputImage)
cv2.imwrite(name_output, XYZMatrix);

# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
