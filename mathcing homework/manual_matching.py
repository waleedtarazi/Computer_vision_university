import cv2 

template_img = cv2.imread('cup.jpg')
test_img = cv2.imread('hidden.jpg')

template_img_bw = cv2.cvtColor(template_img,cv2.COLOR_BGR2GRAY)
test_img_bw = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
  
orb = cv2.ORB_create()
templateKeypoints, templateDescriptors = orb.detectAndCompute(template_img_bw,None)
testKeypoints, testDescriptors = orb.detectAndCompute(test_img_bw,None)
 
# keypoints
matcher = cv2.BFMatcher()
matches = matcher.match(templateDescriptors,testDescriptors)
  
reslut_img = cv2.drawMatches(template_img, templateKeypoints,
test_img, testKeypoints, matches[:30],None)
  
reslut_img = cv2.resize(reslut_img, (1080,720))
 
cv2.imshow("matching results", reslut_img)
cv2.waitKey(0)