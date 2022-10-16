import os
imgfile="D:/talking-head-anime-demo-master/dataSet1/img1/"
labelfile="D:/talking-head-anime-demo-master/dataSet1/label1/"
no=[]
no2=[]
for i in range(0,7135):
    print(i)
    for j in range(2,72):
        try:
            original=imgfile+str(i)+"/MMD"+str(i)+"_"+str(j)+".png"
            if j<10:
                changed=imgfile+str(i)+"/MMD"+str(i)+"M_10"+str(j)+".png"
            else:
                changed=imgfile+str(i)+"/MMD"+str(i)+"M_1"+str(j)+".png"
            os.rename(original,changed)
        except:
            print("pass")

    for j in range(72,142):
        try:
            # original="D:/talking-head-anime-demo-master/dataSet1/img1/"+str(i)+"/MMD"+str(i)+"_"+str(j)+".png"
            # changed="D:/talking-head-anime-demo-master/dataSet1/img1/"+str(i)+"/MMD"+str(i)+"_R"+str(j-70)+".png"
            original=imgfile+str(i)+"/MMD"+str(i)+"_"+str(j)+".png"
            if j-70<10:
                changed=imgfile+str(i)+"/MMD"+str(i)+"R_20"+str(j-70)+".png"
            else:
                changed=imgfile+str(i)+"/MMD"+str(i)+"R_2"+str(j-70)+".png"
            os.rename(original,changed)
        except:
            no.append(i)

for i in range(0,7135):
    print(i)
    for j in range(2,72):
        try:
            original=labelfile+str(i)+"/MMD"+str(i)+"_"+str(j)+".txt"
            if j<10:
                changed=labelfile+str(i)+"/MMD"+str(i)+"M_10"+str(j)+".txt"
            else:
                changed=labelfile+str(i)+"/MMD"+str(i)+"M_1"+str(j)+".txt"
            os.rename(original,changed)
        except:
            print("pass")

    for j in range(72,142):
        try:
            # original="D:/talking-head-anime-demo-master/dataSet1/img1/"+str(i)+"/MMD"+str(i)+"_"+str(j)+".txt"
            # changed="D:/talking-head-anime-demo-master/dataSet1/img1/"+str(i)+"/MMD"+str(i)+"_R"+str(j-70)+".txt"
            original=labelfile+str(i)+"/MMD"+str(i)+"_"+str(j)+".txt"
            if j-70<10:
                changed=labelfile+str(i)+"/MMD"+str(i)+"R_20"+str(j-70)+".txt"
            else:
                changed=labelfile+str(i)+"/MMD"+str(i)+"R_2"+str(j-70)+".txt"
            os.rename(original,changed)
        except:
            no2.append(i)