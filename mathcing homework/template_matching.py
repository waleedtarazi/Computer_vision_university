import cv2 as cv

# cv.TM_SQDIFF is the best for super_side image with 0.2
# cv.TM_CCOEFF_NORMED is the best for half_cup imagea with 0.2
# cv.TM_CCORR not good at all  for two cups 
# cv.TM_CCOEFF_NORMED is good for low_light with 0.3
# hidden works also for 0.2

image = cv.resize(cv.imread('two_cups.jpg',0) , (0,0) ,fx = 0.2 , fy = 0.2)
template =cv.imread("cup.jpg" , 0)
height,width = template.shape
print("image shape is:",image.shape ,"\n" ,"template shape is:",template.shape )
methods = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR,
            cv.TM_CCORR_NORMED, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]
for method in methods:
    image2 = image.copy()
    result = cv.matchTemplate(image2,template,method)
    min_val , max_val , min_loc, max_loc = cv.minMaxLoc(result)
    print(min_loc , max_loc)
    if method in [cv.TM_SQDIFF , cv.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc
    bottom_right = (location[0] + width, location[1] + height)
    cv.rectangle(image2 , location , bottom_right , 255,5)
    cv.imshow('match' , image2)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print(" im happy")







