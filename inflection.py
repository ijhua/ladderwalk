import pandas as pd
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats
from statistics import mean, stdev,median
import cv2
from math import hypot
import datetime
from IPython.display import clear_output

os.system('python3 postprocess.py')


# list of rats
rats = ["MC61","MC87","MC30","MC70","MC45","MC78"]
#define handedness of the rats
right_handed = ["MC45","MC61","MC78","MC87","MC30","MC70"]
left_handed = []

count = 1
#location of h5 files for analysis
folders = []
for rat in rats:
    folders+=glob.glob("/home/ml/Documents/Not_TADSS_Videos/"+rat+"/cut/dlc_output_resnet50/*.h5")
data = []
ind = []
#iterate through every file
for f in folders:
    clear_output(wait=True)
    print(str(count)+"/"+str(len(folders)))
    #read the file
    df = pd.read_hdf(f)
    #define properties
    name=f.split("/")[8]
    run = name.split("_")[2]
    subject = name.split("_")[0]
    date = name.split("_")[1]
    if len(name.split("_")[3])==8:
        crossing=name.split("_")[3][-1]
    elif len(name.split("_")[3])>8:
        crossing = name.split("_")[3][-2:]
    #parameters for later
    likelihood_threshold = 0.1

    #date of injury
    if subject == "MC30":
        date1 = datetime.datetime(2019,11,12)
    elif subject == "MC70":
        date1 = datetime.datetime(2019,3,19)
    elif subject == "MC45":
        date1 = datetime.datetime(2019,7,23)
    elif subject == "MC61":
        date1 = datetime.datetime(2019,6,11)
    elif subject == "MC78":
        date1 = datetime.datetime(2019,4,2)
    elif subject == "MC87":
        date1 = datetime.datetime(2018,12,17)
    week_num = (pd.to_datetime(date).date() - date1.date()).days/7
    #separate weeks into pre and post injury categories
    if week_num <=0:
        week = "Preinjury"
    if week_num>0:
        week="Postinjury"

    #open video in cv2
    video = cv2.VideoCapture(f.split('.')[0]+"_labeled.mp4")
    #find framerate
    framerate = video.get(cv2.CAP_PROP_FPS)

    #open the rung hdf5s
    path = f.split(".")[0].split("/")[:-2]+["dlc_rung_r50"]
    rung_folder = os.path.join(*(path))
    rung_file= f.split(".")[0].split("/")[-1].split("_")[0]+"_"+f.split(".")[0].split("/")[-1].split("_")[1]+"_"+f.split(".")[0].split("/")[-1].split("_")[2]+"_"+f.split(".")[0].split("/")[-1].split("_")[3]+"_"+f.split(".")[0].split("/")[-1].split("_")[4]+"_"+f.split(".")[0].split("/")[-1].split("_")[5]
    #read the file with the rung data
    rung_df = pd.read_hdf("/"+rung_folder+"/"+rung_file+"_"+"DLC_resnet50_LadderWalkMar12shuffle1_450000.h5")
    rung_df = rung_df['DLC_resnet50_LadderWalkMar12shuffle1_450000']
    #make a list of all the rungs from 1 to 62. There are too many to do manually like with the limbs
    rung_list = []
    for i in range(1,63):
        rung_list.append("rung_"+str(i))
    #filter each column of the rung dataframe based on likelihood
    for rung in rung_list:
        rung_df[rung]=likelihood_filter(rung_df[rung],0.8, fill=False)
    #get the mean and standard error for of the rungs
    rung_mean = rung_df.agg(["mean","sem"])
    #remove any column with NaN values
    rung_mean = rung_mean.dropna(axis='columns')
    rung_shell = rung_mean.drop(['y','likelihood'],axis=1,level=1)
    #make empty numpy arrays the same size as the column
    rung_x = np.empty(shape=(int(rung_shell.shape[1]),1))
    rung_y = np.empty(shape=(int(rung_shell.shape[1]),1))
    #split the dataframe columns into the numpy arrays so we have all the x together and all the y together
    #the list of column names is not in order, but it shouldn't matter as long as the x and y line up in position
    for rung in rung_shell.columns.get_level_values(0):
        key = rung
        num = rung_shell.columns.get_loc(key)
        rung_x[num]=rung_mean[key]["x"]["mean"]
        rung_y[num]=rung_mean[key]["y"]["mean"]
    #remove coordinates with outliers in the y direction
    rung_x = rung_x[not_outliers(rung_y)]
    rung_y = rung_y[not_outliers(rung_y)]

    #estimate distance by selecting 3 groups of 5 rungs
    r1 = rung_shell.iloc[0,4:11][0]
    r2 = rung_shell.iloc[0,4:11][1]
    r3 = rung_shell.iloc[0,4:11][2]
    r4 = rung_shell.iloc[0,4:11][3]
    r5 = rung_shell.iloc[0,4:11][4]

    r6 = rung_shell.iloc[0,21:26][0]
    r7 = rung_shell.iloc[0,21:26][1]
    r8 = rung_shell.iloc[0,21:26][2]
    r9 = rung_shell.iloc[0,21:26][3]
    r10 = rung_shell.iloc[0,21:26][4]

    r11 = rung_shell.iloc[0,30:35][0]
    r12 = rung_shell.iloc[0,30:35][1]
    r13 = rung_shell.iloc[0,30:35][2]
    r14 = rung_shell.iloc[0,30:35][3]
    r15 = rung_shell.iloc[0,30:35][4]

    #calculate the x distance between adjacent rungs
    d1 = (r2-r1)
    d2 = (r3-r2)
    d3 = (r4-r3)
    d4 = (r5-r4)

    d5 = (r7-r6)
    d6 = (r8-r7)
    d7 = (r9-r8)
    d8 = (r10-r9)

    d9 = (r12-r11)
    d10 = (r13-r12)
    d11 = (r14-r13)
    d12 = (r15-r14)

    #make array of the differences. remove outliers. find the median
    rung_dists = np.array([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12])
    rung_dists = rung_dists[not_outliers(rung_dists)[0]]
    rung_median = median(rung_dists)

    #position data
    #left crossings
    if run[0] == "L":
        #apply handedness
        if subject in right_handed:
            limb_front = "Nondominant Front"
            limb_back = "Nondominant Back"
        elif subject in left_handed:
            limb_front = "Dominant Front"
            limb_back = "Dominant Back"
        #split all the limbs
        #front left
        df_fingers = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"left fingers")
        #back left
        df_toes = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"left toes")

    #right crossings
    if run[0] == "R":
        #apply handedness
        if subject in left_handed:
            limb_front = "Nondominant Front"
            limb_back = "Nondominant Back"
        elif subject in right_handed:
            limb_front = "Dominant Front"
            limb_back = "Dominant Back"
        #split all the limbs
        #front right
        df_fingers = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"right fingers")
        #back right
        df_toes = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"right toes")

    #filter dataframes by likelihood
    #front
    df_fingers = likelihood_filter(df_fingers,likelihood_threshold)
    df_fingers.name = limb_front

    #back
    df_toes = likelihood_filter(df_toes,likelihood_threshold)
    df_toes.name = limb_back
    dfs = [df_fingers,df_toes]
    #calculate the second derivative of position to give acceleration.
    #The peaks in the acceleration correspond to points of inflection in the velocity, which is where there the position graph will change direction (or when the rat puts its paw down, then lifts it up)
    for df in dfs:
        limb = df.name
        second_x = np.gradient(np.gradient(df.x))
        pos_peak_x = find_peaks(second_x,prominence=5,width=1,height=10)[0]
        neg_peak_x = find_peaks(-second_x,prominence=5,width=1,height=10)[0]
        peak_x = np.append(pos_peak_x,neg_peak_x)
        peak_x = np.sort(peak_x)

        #plot graph of the acceleration and peaks
        plt.close()
        plt.plot(df.x,second_x)
        plt.scatter(df.x.iloc[peak_x],second_x[peak_x])
        plt.xlabel('x')
        plt.ylabel('x acceleration')
        plt.title(subject + " " + date + " "+ run + " "+ limb + " Second Derivative Peaks")
        plt.savefig("/home/ml/Documents/methods_figures/inflection/"+subject+"_"+date+"_"+run+"_"+limb+"_"+"poi.png")

        #plot graph of position with points at the points of inflection
        plt.close()
        plt.plot(df.x,df.y)
        plt.scatter(df.x.iloc[peak_x],df.y.iloc[peak_x])
        plt.gca().invert_yaxis()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(subject + " " + date + " "+ run + " "+ limb + " Position Windows")
        plt.savefig("/home/ml/Documents/methods_figures/inflection/"+subject+"_"+date+"_"+run+"_"+limb+"_"+"window.png")
        plt.close()
    print("Graphs done")

    for df in dfs:
        limb = df.name
        #second derivative
        second_x = np.gradient(np.gradient(df.x))
        #positive peaks (index)
        pos_peak_x = find_peaks(second_x,prominence=5,width=1,height=10)[0]
        #negative peaks (index)
        neg_peak_x = find_peaks(-second_x,prominence=5,width=1,height=10)[0]
        #combine the arrays of indicies
        peak_x = np.append(pos_peak_x,neg_peak_x)
        #sort arrays
        peak_x = np.sort(peak_x)
        #make empty multi-index dataframe for the step number and the coordinates of the step
        new_index = pd.MultiIndex(levels=[[],[]],codes=[[],[]],names=["step","coords"])
        new_df = pd.DataFrame(columns=new_index)
        #videos had a different number of steps so this needed to be flexible
        #define the swing phase to be between adjacent peaks
        a=0
        for i,k in zip(peak_x[0::2], peak_x[1::2]):
            #take all points between adjacent peaks
            df2 = df.iloc[range(i,k+1)]
            df2 = df2.drop("likelihood",axis=1)
            #make the x and y positions relative to the first point of the step
            df2.x = df2.x - df2.x.iloc[0]
            df2.y = df2.y - df2.y.iloc[0]
            df2 = df2.reset_index()
            if a>0:
                #some steps have a different number of points from the first step, so this adds rows to the new dataframe until it has enough
                while len(df2)>len(new_df):
                    new_df = new_df.append([[]]).reset_index(drop=True)
            #add this step to the dataframe
            new_df['step_'+str(a),'x']= df2.x
            new_df['step_'+str(a),'y']= df2.y
            #advance counter
            a+=1
        #calculate the parameters of interest
        if len(new_df)>0:
            num_windows = len(new_df.columns.levels[0])
            #avg distance and velocity over the step
            dist_x = []
            dist_y = []
            vels_x = []
            vels_y = []
            all_t = []
            for col in new_df.columns.levels[0]:
                df3 = new_df[col]
                df3 = df3.reset_index(drop=True).dropna()
                #distance between 2 adjacent rungs = 0.7cm
                #conversion: 1 pixel = 0.7/median centimeters
                dx = abs(df3['x'][0] - df3['x'][len(df3)-1])*(0.7/rung_median)
                dy = abs(df3['y'][0] - df3['y'][len(df3)-1])*(0.7/rung_median)
                dt = len(df3)*(1/framerate)
                vx = dx/dt
                vy = dy/dt
                d = hypot(dx,dy)
                v = hypot(vx,vy)
                ind.append([subject,date,week,run,limb,framerate,col,dt,dx,dy,d,vx,vy,v])
    count+=1

ind_df = pd.DataFrame(ind,columns=["subject","date","week","run","limb","framerate","step_no","time","dx","dy","distance","vx","vy","velocity"])
ind_df["date"] = pd.to_datetime(ind_df["date"])
ind_df.to_csv("/home/ml/Documents/individual_steps.csv")

calcs = []
for index,row in ind_df.iterrows():
    #get the subject ID
    subject = row['subject']
    week = row['week']
    #definethe limb column
    limb = row['limb']

    t = row["time"]
    dx = row["dx"]
    d = row["distance"]
    vx = row["vx"]
    v = row["velocity"]

    calcs.append([subject,week,limb,t,dx,d,vx,v])
calc_df = ind_df.loc[ind_df,["subject","week","run","limb","time","dx","distance","vx","velocity"]]
calc_df = calc_df.rename({"time":"t","distance":"d","velocity":"v"})

#trends (calculate mean and sem for each parameter)
trend = calc_df.groupby(["week","limb"])["t","dx","d","vx","v"].agg(["mean","sem"])
trend = trend.reset_index()
trend.to_csv("/home/ml/Documents/avg_of_all_steps.csv")

fd = trend.loc[trend["limb"]=="Dominant Front"]
bd = trend.loc[trend["limb"]=="Dominant Back"]
fn = trend.loc[trend["limb"]=="Nondominant Front"]
bn = trend.loc[trend["limb"]=="Nondominant Back"]

#plot graphs of pre and post injury for each of the calculated averages
limbs = [fd,bd,fn,bn]
for limb in limbs:
    limb = limb.reset_index()
    name = limb["limb"][0]

    plt.close()
    plt.errorbar(limb[('week', '')],limb[('t', 'mean')],yerr=limb[('t', 'sem')],uplims=True, lolims=True)
    plt.xlabel("Week")
    plt.ylabel("Time (sec)")
    plt.title(name + " Avg Time Per Step")
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/loc_trends/"+name+"_avg_time.png")

    plt.close()
    plt.errorbar(limb[('week', '')],limb['d', 'mean'],yerr=limb[('d', 'sem')],uplims=True, lolims=True)
    plt.xlabel("Week")
    plt.ylabel("Distance (cm)")
    plt.title(name + " Avg Step Distance")
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/loc_trends/"+name+"_avg_distance.png")

    plt.close()
    plt.errorbar(limb[('week', '')],limb[('dx', 'mean')],yerr=limb[('dx', 'sem')],uplims=True, lolims=True)
    plt.xlabel("Week")
    plt.ylabel("Distance (cm)")
    plt.title(name + " Avg x Component of Step Distance")
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/loc_trends/"+name+"_avg_x_distance.png")

    plt.close()
    plt.annotate(str(round(limb['t_v'][0],5)),("Postinjury",limb[('v', 'mean')].max()))
    plt.errorbar(limb[('week', '')],limb[('v', 'mean')],yerr=limb[('v', 'sem')],uplims=True, lolims=True)
    plt.xlabel("Week")
    plt.ylabel("Velocity (cm/sec)")
    plt.title(name + " Avg Step Velocity")
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/loc_trends/"+name+"_avg_velocity.png")

    plt.close()
    plt.annotate(str(round(limb['t_vx'][0],5)),("Postinjury",limb[('vx', 'mean')].max()))
    plt.errorbar(limb[('week', '')],limb[('vx', 'mean')],yerr=limb[('vx', 'sem')],uplims=True, lolims=True)
    plt.xlabel("Week")
    plt.ylabel("Velocity (cms/sec)")
    plt.title(name + " Avg x Component of Step Velocity")
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/loc_trends/"+name+"_avg_x_velocity.png")

    plt.close()
print("Done")
