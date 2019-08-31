#Created By: nishantKr18
##################################
#     #  #            #   ##      
# #   #  #            #  #  #     
#  #  #  # ##  # ##   #   ##      
#   # #  ##    ##     #  #  #     
#    ##  # ##  #      #   ##      
################################## 

import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from statistics import mean
from sympy import sympify
from time import sleep
from sympy import Symbol, sympify, factor, plot, solve, sin, Limit, Derivative, init_printing
from sympy import Integral, log

p = Symbol('p')
y = Symbol('y')
z = Symbol('z')

# cp = cv2.VideoCapture(0)

def annotate(frame, label, location = (20,30)):
    cv2.putText(frame, label, location, cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.5,
                color = (255, 255, 0),
                thickness =  1,
                lineType =  cv2.LINE_AA)
def extract_digit(gray_frame, rect, pad = 10):
    x, y, w, h = rect
    cropped_digit = gray_frame[y-pad:y+h+pad, x-pad:x+w+pad]
    cropped_digit = cropped_digit/255.0

    #only look at images that are somewhat big:
    if cropped_digit.shape[0] >= 20 and cropped_digit.shape[1] >= 20:
        temp = blank_image = np.zeros((45,45), float)
        temp.fill(1)
        mask = cropped_digit

        if mask.shape[0] >= mask.shape[1]:
            j = int(45 * mask.shape[1] / mask.shape[0])
            i = 45
        else:

            i = int(45 * mask.shape[0] / mask.shape[1])
            j = 45

        mask = cv2.resize(mask, (j, i))
        if i == 45:
            jt = int((45 - j)/2)
            temp[0:i, jt:jt+j] = mask
        else:
            it = int((45 - i)/2)
            temp[it:it+i, 0:j] = mask
        # temp = cv2.resize(temp, (150, 150), interpolation=cv2.INTER_AREA)
        # cropped_digit = cv2.resize(cropped_digit, (45, 45))
    else:
        return
    return temp
def img_to_gray(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, blockSize = 321, C = 28)
    gray_img = cv2.bitwise_not(gray_img)
    return gray_img
def showPlot(frame_list):
    fig=plt.figure()
    size = int(len(frame_list)/2)
    for i in range(len(frame_list)):
        img = frame_list[i]
        fig.add_subplot(size, size, i+1)
        plt.imshow(img)
    plt.show()
def solve2(equation, perform):
    global labelz
    current = 0
    equation2 = []
    TypeOfEquation = 0
    flag = False
    for i in equation:
        if(i<10):
            flag = True
            current = current*10+i
        else:
            if(flag == True):equation2.append(current)
            flag = False
            equation2.append(labelz[i])
            current = 0
    if(flag == True):equation2.append(current)
    print(equation2)
    print()

    if ('int' in equation2):TypeOfEquation = 1
    elif('d' in equation2):TypeOfEquation = 2
    elif('lim' in equation2):TypeOfEquation = 3
    elif(('p'or'y'or'z') in equation2):TypeOfEquation = 4
    else:TypeOfEquation = 5

    s = ''
    for i in equation2: s+=str(i)
    # print(TypeOfEquation)
    print('Recognized equation is :'+s)

    if(perform==False):return

    print()
    print('The Answer is :')
    if(TypeOfEquation == 1): 
        a = Integral(sympify(s[3:])).doit()
        print(str(a)+'\n\n'+'The Graph:')
        plot(a)
    elif(TypeOfEquation == 2): 
        a = Derivative(sympify(s[1:])).doit()
        print(str(a)+'\n\n'+'The Graph:')
        plot(a)
    elif(TypeOfEquation == 3): 
        if(s[6:8]=='oo'):a = Limit(sympify(s[8:]), sympify(s[3]), sympify(s[6:8])  ).doit()
        else : a = Limit(sympify(s[7:]), sympify(s[3]), sympify(s[6])  ).doit()
        print(a)
    elif(TypeOfEquation == 4):
        expr = sympify(s)
        print('factorized Expression => ', factor(expr))
        print('on Solving, We get => ',solve(expr))
        print('The Graph of The Function :')
        plot(expr)
    elif(TypeOfEquation == 5):
        print(sympify(s))
        print('On further simplification => ', sympify(s).evalf())


    # return sympify(s)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("loading model")
model = load_model("full_model_onMyData.mnist")
labelz = dict(enumerate(['0', '1','2','3','4','5','6','7','8','9','-','+','*','(',')','cos','d','E','oo','int','lim','log','p','pi','rA','sin','sqrt','tan','y','z']))

def process(frame, perform=True):
    gray_frame = img_to_gray(frame)
    contours,_ = cv2.findContours(cv2.bitwise_not(gray_frame.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(contour) for contour in contours]
    rects.sort(key=lambda x: x[0]) #sorts the rects for left to right(based on x) {rects has all the detected bounding boxes in x,y,w,h form}
    rects = np.asarray(rects)
    if(len(rects)>0): rects = [rect for rect in rects if rect[2] > 0.1*mean(rects[:,2]) and rect[3] > 0.1*mean(rects[:,3]) ]
    equation = []
    mnist_frame_list = []
    for rect in rects:
        x, y, w, h = rect
        pad = 10
        mnist_frame = extract_digit(gray_frame, rect, pad = pad)
        if mnist_frame is not None:
            mnist_frame_list.append(mnist_frame)
            mnist_frame = mnist_frame.reshape(-1, 45, 45, 1)
            class_prediction = model.predict_classes(mnist_frame, verbose = False)[0]
            # prediction = np.around(np.max(model.predict(mnist_frame, verbose = False)), 2)
            # label = str(prediction) # if you want probabilities
            equation.append(class_prediction)
            cv2.rectangle(frame, (x - pad, y - pad), (x + pad + w, y + pad + h),
                          color = (255, 255, 0))

            label = labelz[class_prediction]

            annotate(frame, label, location = (rect[0], rect[1]))

    # cv2.imshow('gray_frame', gray_frame)
    # cv2.imshow('frame', frame)
    # key = cv2.waitKey(0)

    # plt.imshow(gray_frame)
    # plt.show()
    plt.imshow(frame)
    plt.show()
    
    showPlot(mnist_frame_list)
    solve2(equation, perform)

if __name__ == '__main__':
    # ret, frame = cp.read()
    frame = cv2.imread('Test_images/555diff.png')
    process(frame)