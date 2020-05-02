import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob, os
import datetime as dt
import cv2
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import stats

os.system('python3 /home/ml/Documents/ladderwalk/postprocess.py')


# list of rats
rats = ["MC61","MC87","MC30","MC70","MC45","MC78"]
#define handedness of the rats
right_handed = ["MC45","MC61","MC78","MC87","MC30","MC70"]
left_handed = []

#location of h5 files for analysis
folders = []
for rat in rats:
    folders+=glob.glob("/home/ml/Documents/Not_TADSS_Videos/"+rat+"/cut/dlc_output_resnet50/*.h5")

#set up dataframe for future
scores = []
excluded_scores =[]
score_cols = ["subject", "date", "run", "crossing","limb","comp_hits","comp_misses","comp_steps"]

#iterate through every file
for f in folders:
    #read the file
    df = pd.read_hdf(f)
    #define properties
    name=f.split("/")[8]
    run = name.split("_")[2]
    subject = name.split("_")[0]
    date = name.split("_")[1]
    if len(name.split("_")[3])==9:
        crossing=name.split("_")[3][-1]
    elif len(name.split("_")[3])>9:
        crossing = name.split("_")[3][-2:]
    #parameters for later
    likelihood_threshold = 0.1
    xheight = 20
    xdist = 4
    yheight = 5
    ydist = 5
    zero_threshold = 5

    #incorporate rung information that matches the current file
    #set rung file path
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
        df_wrist = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"left wrist")
        df_fingers = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"left fingers")
        #back left
        df_ankle = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"left ankle")
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
        df_wrist = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"right wrist")
        df_fingers = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"right fingers")
        #back right
        df_ankle = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"right ankle")
        df_toes = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"right toes")

    #filter dataframes by likelihood
    #front
    df_wrist = likelihood_filter(df_wrist,likelihood_threshold)
    df_fingers = likelihood_filter(df_fingers,likelihood_threshold)

    #back
    df_ankle = likelihood_filter(df_ankle,likelihood_threshold)
    df_toes = likelihood_filter(df_toes,likelihood_threshold)

    #get the x velocity peaks using clusters


    fingers_forward_list = inflection(df_fingers)

    toes_forward_list = inflection(df_toes)


    front_forward = fingers_forward_list

    back_forward = toes_forward_list

    #figure out the y position threshold for peakfinding
    y_pos_threshold_front,y_pos_threshold_back = zero_velocity_y_position()

    fingers_slip_list = find_y_position_peaks(df_fingers,y_pos_threshold_front,ydist)
    wrist_slip_list = find_y_position_peaks(df_wrist,y_pos_threshold_front,ydist)

    ankle_slip_list = find_y_position_peaks(df_ankle,y_pos_threshold_back,ydist)
    toes_slip_list = find_y_position_peaks(df_toes,y_pos_threshold_back,ydist)


    #using only 1 point on each limb appears to be more accurate than using the union of two points
    front_slip = fingers_slip_list#peak_list_union(wrist_slip_list,fingers_slip_list)
    #join back lists for the two points on the hindlimb
    back_slip = toes_slip_list#peak_list_union(ankle_slip_list,toes_slip_list)

    #count the number of steps, hits and slips for left and right crossings only on the visible side.
    if run[0] == "L":
        #number of x peaks is the number of total steps
        total_steps_fl = (len(front_forward))//2
        total_steps_bl = (len(back_forward))//2
        total_steps_fr = np.nan
        total_steps_br = np.nan

        slip_count_fl = len(front_slip)
        slip_count_bl = len(back_slip)
        slip_count_fr = np.nan
        slip_count_br = np.nan

        hit_count_fl = total_steps_fl - slip_count_fl
        hit_count_bl = total_steps_bl - slip_count_bl
        hit_count_fr = np.nan
        hit_count_br = np.nan
    if run[0] == "R":
        #number of x peaks is the number of total steps
        total_steps_fl = np.nan
        total_steps_bl = np.nan
        total_steps_fr = (len(front_forward))//2
        total_steps_br = (len(back_forward))//2

        slip_count_fl = np.nan
        slip_count_bl = np.nan
        slip_count_fr = len(front_slip)
        slip_count_br = len(back_slip)

        hit_count_fl = np.nan
        hit_count_bl = np.nan
        hit_count_fr = total_steps_fr - slip_count_fr
        hit_count_br = total_steps_br - slip_count_br
    #define lists for each limb for all of the scores that will be a row in the final dataframe
    score_front_l = [subject,date,run,crossing,limb_front,hit_count_fl,slip_count_fl,total_steps_fl]
    score_back_l = [subject,date,run,crossing,limb_back,hit_count_bl,slip_count_bl,total_steps_bl]
    score_front_r = [subject,date,run,crossing,limb_front,hit_count_fr,slip_count_fr,total_steps_fr]
    score_back_r = [subject,date,run,crossing,limb_back,hit_count_br,slip_count_br,total_steps_br]
    #put each of those lists into the larger list that we made a the start
    if hit_count_fl<=0:
        excluded_scores.append(score_front_l)
    else:
        scores.append(score_front_l)
    if hit_count_bl<=0:
        excluded_scores.append(score_back_l)
    else:
        scores.append(score_back_l)
    if hit_count_fr<=0:
        excluded_scores.append(score_front_r)
    else:
        scores.append(score_front_r)
    if hit_count_br<=0:
        excluded_scores.append(score_back_r)
    else:
        scores.append(score_back_r)

#make the list into a dataframe with all of the scores
score_df = pd.DataFrame(scores,columns=score_cols)
#make the date into a datetime format
score_df["date"] = pd.to_datetime(score_df["date"])

#open the file of manual scores
test_human = pd.read_csv("/home/ml/Documents/updated_human_scores.csv")
#date column into datetime format
test_human['date'] = pd.to_datetime(test_human['date'])

#merge the computational and human score dataframes
all_score = score_df.merge(test_human,on=["subject","date","run","limb"])
#drop the empty rows (should just be the rows with the side of the rat that is further away)
all_score = all_score.dropna(axis=0)

all_score.to_csv("/home/ml/Documents/methods_comparison_RN50_inflection.csv")

exclude_df  = pd.DataFrame(excluded_scores,columns=score_cols)
exclude_df["date"] = pd.to_datetime(exclude_df["date"])

comb_ex_df = exclude_df.merge(test_human,on=["subject","date","run","limb"])
comb_ex_df = comb_ex_df.dropna(axis=0)
comb_ex_df.to_csv("/home/ml/Documents/methods_excluded_inflection.csv")

print("Score calculation and comparison done.")
