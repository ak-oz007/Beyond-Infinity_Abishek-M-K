import cv2
from modules import handshake
from modules import running
from modules import kicking
from modules import punching
from modules import push
from modules import walking
from modules import posedector as pdr

###

Nature = {'anger' : 0,'threat' : 0,'fight' :0,'OK-OK' : 0,'normal': 0}



thresh = 0.50 # Threshold to detect object
# cap = cv2.VideoCapture("D:\\AEEE_THINGS\\Sem_5\\Projects\\Hackathon\\code\\video\\video\\51_9_4.avi")
#cap = cv2.VideoCapture("D:\\AEEE_THINGS\\Sem_5\\Projects\\Hackathon\\code\\video\\video\\52_9_0.avi")
#cap = cv2.VideoCapture("D:\\AEEE_THINGS\\Sem_5\\Projects\\Hackathon\\code\\video\\video\\_I_ll_Just_Wave_My_Hand__sung_by_Cornerstone_Male_Choir_wave_u_cm_np1_fr_med_0.avi")
cap = cv2.VideoCapture("D:\\AEEE_THINGS\\Sem_5\\Projects\\Hackathon\\code\\video\\A_Beautiful_Mind_4_stand_f_cm_np1_fr_med_13.avi")
#cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



class_names = []
class_file = 'coco.names'
with open(class_file,'rt')as f:
    class_names = f.read().strip("\n").split('\n')

configpath =  'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightpath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightpath,configpath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# classIds, confidence, bounding_box = net.detect(img,confThreshold=0.5)
while cap.isOpened():
    success, img = cap.read()
    if not success:
        break
    classIds, confidence, bounding_box = net.detect(img,confThreshold=thresh)
    print(classIds,bounding_box)
    count = 0
    list_of_all_objects = {}
    if len(classIds) !=0:
        for classId, confidenc, box in zip(classIds.flatten(),confidence.flatten(),bounding_box):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,class_names[classId - 1].upper(),(box[0]+50,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) # for printing the class name
            cv2.putText(img, str(round(confidenc * 100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0), 2) #for printing the prediction accuracy
            cv2.imshow("output", img)

            if class_names[classId - 1] not in list_of_all_objects:
                list_of_all_objects[class_names[classId-1]] = 1
            else:
                list_of_all_objects[class_names[classId-1]] +=1


            if(class_names[classId -1] == 'person') and str(round(confidenc *100 , 2) > 50):
                try:
                    if handshake.handshake(cap) or walking.walks(cap):
                        Nature['normal'] = 1
                        Nature['anger'] = 0
                        Nature['fight'] = 0
                        Nature['OK-OK'] = 0
                        Nature['threat'] = 0
                        print("Handshake : ",end=" ")
                        print(handshake.handshake(cap))
                        print("\nWalking : ",end=" ")
                        print(walking.walks(cap))
                    if kicking.kicking(cap) or punching.punching(cap):
                        Nature['normal'] = 0
                        Nature['anger'] = 1
                        Nature['fight'] = 1
                        Nature['OK-OK'] = 0
                        Nature['threat'] = 1
                        print("Kicked : ",end=" ")
                        print(kicking.kicking(cap))
                        print("\n Punched : ",end=" ")
                        print(punching.punching(cap))
                        # print(kicking.kicking(cap) or punching.punching(cap))
                    if push.push(cap) or running.running(cap):
                        Nature['normal'] = 0
                        Nature['anger'] = 0
                        Nature['fight'] = 0
                        Nature['OK-OK'] = 1
                        Nature['threat'] = 1
                        print("Pushed : ",end=" ")
                        print(push.push(cap))
                        print("\nRunning :",end=" ")
                        print(running.running(cap))
                except:
                    pass

                if (Nature['fight'] or Nature['anger']) or Nature['threat'] == 1:
                    pdr.posedetector(cap)
                else:
                    pass

    print("\nThe total objects with count  = ",end =" \n")
    print(list_of_all_objects)

    cv2.waitKey(1)
    cap.release()