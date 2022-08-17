import cv2,time,pandas
from datetime import datetime

background=None

statues_list=[None,None]
motion=[]
video=cv2.VideoCapture(0)
video.read()
time.sleep(1.0)
df=pandas.DataFrame(columns=["Start","End"])

while True:


    check,frame=video.read()
    statues=0
    #print(check)
    #print(frame)

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)
    if background is None:
        background=gray
        continue

    delta=cv2.absdiff(background,gray)
    threshold=cv2.threshold(delta,30,255,cv2.THRESH_BINARY)[1]
    threshold=cv2.dilate(threshold,None,iterations=5)
    (cnts,_)=cv2.findContours(threshold.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for a in cnts:
        if cv2.contourArea(a) < 1000:
            continue
        statues=1

        (x,y,h,w)=cv2.boundingRect(a)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    statues_list.append(statues)
    if statues_list[-1]==1 and statues_list[-2]==0:
        motion.append(datetime.now())

    if statues_list[-1]==0 and statues_list[-2]==1:
        motion.append(datetime.now())

    #cv2.imshow("capturing",gray)
    #cv2.imshow("Shit",delta)
    #cv2.imshow("Thresh",threshold)
    cv2.imshow("Detect",frame)
    key= cv2.waitKey(1)


    if key==ord('q'):
        if statues==1:
            motion.append(datetime.now())
        break


print(statues_list)
print(motion)
for i in range(0,len(motion),2):
    df=df.append({"Start":motion[i],"End":motion[i+1]},ignore_index=True)

df.to_csv("Motions.csv")
video.release()
cv2.destroyAllWindows
