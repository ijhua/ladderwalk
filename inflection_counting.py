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
rats = ["MC61","MC87","MC30","MC70","MC78"]
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
score_cols = ["subject", "date","week_number","injury", "run", "crossing","limb","comp_hit","comp_miss","comp_steps","comp_error_rate"]

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

    if subject == "MC30":
        date1 = dt.datetime(2019,11,12)
    elif subject == "MC70":
        date1 = dt.datetime(2019,3,19)
    elif subject == "MC45":
        date1 = dt.datetime(2019,7,23)
    elif subject == "MC61":
        date1 = dt.datetime(2019,6,11)
    elif subject == "MC78":
            date1 = dt.datetime(2019,4,2)
    elif subject == "MC87":
        date1 = dt.datetime(2018,12,17)
    week_num = (pd.to_datetime(date).date() - date1.date()).days/7
    week_round = round(week_num,0)
    if week_num <=0:
        week_cat = "Preinjury"
    if week_num>0:
        week_cat="Postinjury"

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

        error_fl = slip_count_fl/total_steps_fl*100
        error_bl = slip_count_bl/total_steps_bl*100
        error_fr = np.nan
        error_br = np.nan

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

        error_fl = np.nan
        error_bl = np.nan
        error_fr = slip_count_fr/total_steps_fr*100
        error_br = slip_count_br/total_steps_br*100

    #define lists for each limb for all of the scores that will be a row in the final dataframe
    score_front_l = [subject,date,week_round,week_cat,run,crossing,limb_front,hit_count_fl,slip_count_fl,total_steps_fl,error_fl]
    score_back_l = [subject,date,week_round,week_cat,run,crossing,limb_back,hit_count_bl,slip_count_bl,total_steps_bl,error_bl]
    score_front_r = [subject,date,week_round,week_cat,run,crossing,limb_front,hit_count_fr,slip_count_fr,total_steps_fr,error_fr]
    score_back_r = [subject,date,week_round,week_cat,run,crossing,limb_back,hit_count_br,slip_count_br,total_steps_br,error_br]
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
test_human = pd.read_csv("/home/ml/Downloads/LW_Manual_scores_for_ICC_2020-05-20.csv")
#date column into datetime format
test_human['date'] = pd.to_datetime(test_human['date'])
test_human["avg_human_hit"] = test_human[["human_hit_1","human_hit_2"]].mean(axis=1)
test_human["avg_human_miss"] = test_human[["human_miss_1","human_miss_2"]].mean(axis=1)
test_human["avg_human_steps"] = test_human[["human_steps_1","human_steps_2"]].mean(axis=1)
test_human["avg_human_error_rate"] = test_human["avg_human_miss"]/test_human["avg_human_steps"]*100
#merge the computational and human score dataframes
all_score = score_df.merge(test_human,on=["subject","date","run","limb","injury"])
#drop the empty rows (should just be the rows with the side of the rat that is further away)
all_score = all_score.dropna(axis=0)

all_score.to_csv("/home/ml/Documents/allH_comp_scores_"+dt.datetime.today().strftime('%Y-%m-%d')+".csv")

exclude_df  = pd.DataFrame(excluded_scores,columns=score_cols)
exclude_df["date"] = pd.to_datetime(exclude_df["date"])

comb_ex_df = exclude_df.merge(test_human,on=["subject","date","run","limb",'injury'])
comb_ex_df = comb_ex_df.dropna(axis=0)
comb_ex_df.to_csv("/home/ml/Documents/excluded_crossings_inflection.csv")

print("Score calculation and comparison done. Working on graphs")
'''


calcs=[]
#interate through every row in the dataframe.
#this way is slower than just subtracting columns, but it allows us to set the date of injury as 0
#TODO: make this section more efficient with time and maybe just make it a dict or datafraem for when there are more rats to deal with
for index,row in all_score.iterrows():
    #get the subject ID
    subject = row['subject']
    week = row['week_category']
    limb = row['limb']
    #only calculate the computational score if the number of steps is not 0
    if row["comp_steps"] != 0:
        comp_score = row["comp_misses"]/row["comp_steps"]*100
    else:
        comp_score = np.nan
    #define the main calculations between columns
    comp_steps = row["comp_steps"]
    comp_slips = row["comp_misses"]
    comp_hits = row['comp_hits']
    human_score = row["human_miss"]/row["human_steps"]*100
    human_steps = row["human_steps"]
    human_miss = row["human_miss"]
    human_hits = row["human_hit"]
    #append to the list
    calcs.append([subject,week,limb,comp_score,comp_steps,comp_slips,comp_hits,human_score,human_steps,human_miss,human_hits])
#change all the calculations into a dataframe
calc_df = pd.DataFrame(calcs,columns=["subject","week","limb","comp_score","comp_steps","comp_misses","comp_hits","human_score","human_steps","human_misses","human_hits"])
#could round the number of weeks (more useful when there are more than 2 relevant weeks of data)
#calc_df = calc_df.round({"week":0})
#drop all rows with any nan values
calc_df = calc_df.dropna(axis=0)

#for non-averaged data (to make histogram and pairwise comparisons)
new_calc = calc_df
#calculate the new columns
new_calc["step_diff"] = new_calc["comp_steps"]-new_calc["human_steps"]
new_calc["miss_diff"] = new_calc["comp_misses"]-new_calc["human_misses"]
new_calc["hit_diff"] = new_calc["comp_hits"]-new_calc["human_hits"]

#make dataframes for each limb
cfd = new_calc.loc[new_calc["limb"] == "Dominant Front"]

cfn = new_calc.loc[new_calc["limb"] =="Nondominant Front"]

cbd = new_calc.loc[new_calc["limb"] =="Dominant Back"]

cbn = new_calc.loc[new_calc["limb"] =="Nondominant Back"]

#list of dataframes separated by limb
calc_limbs = [cfd,cfn,cbd,cbn]

#make 3 graphs for each limb
#TODO: change the titles to be less jargony
for limb in calc_limbs:
    limb = limb.reset_index()
    #name of limb to go in graph title
    name = limb["limb"][0]

    #Difference in steps
    plt.close()
    plt.hist(limb["step_diff"],label='Multipoint Difference')
    plt.legend()
    plt.title("Difference in number of steps "+name)
    plt.xlabel("Computation - Human")
    plt.ylabel("Number of Runs")
    plt.savefig("/home/ml/Documents/methods_figures/histograms_inflection/steps_"+name+'.png')

    #difference in misses
    plt.close()
    plt.hist(limb["miss_diff"],label='Multipoint Difference')
    plt.legend()
    plt.title("Difference in number of misses "+name)
    plt.xlabel("Computation - Human")
    plt.ylabel("Number of Runs")
    plt.savefig("/home/ml/Documents/methods_figures/histograms_inflection/misses_"+name+'.png')

    #difference in hits
    plt.close()
    plt.hist(limb["hit_diff"],label='Multipoint Difference')
    plt.legend()
    plt.title("Difference in number of hits "+name)
    plt.xlabel("Computation - Human")
    plt.ylabel("Number of Runs")
    plt.savefig("/home/ml/Documents/methods_figures/histograms_inflection/hits_"+name+'.png')
    plt.close()

#new dataframe that calculates the mean of each in that list below
df_new = calc_df.groupby(["week","limb"])["comp_score","comp_steps","comp_misses","human_score","human_steps","human_misses"].agg(["mean",'sem'])

df_new=df_new.reset_index()
df_new = df_new.sort_values(by=["week"])

#separate dataframe by limb
fd = df_new.loc[df_new["limb"] == "Dominant Front"]

fn = df_new.loc[df_new["limb"] =="Nondominant Front"]

bd = df_new.loc[df_new["limb"] =="Dominant Back"]

bn = df_new.loc[df_new["limb"] =="Nondominant Back"]

#list of limbs
limbs = [fd,fn,bd,bn]

#graphs: average of each
#make 3 graphs per limb: percent slip, number of steps, number of Slips
#TODO: make titles less jargony
for limb in limbs:
    limb = limb.reset_index()
    name = limb["limb"][0]

    plt.close()
    plt.figure()
    plt.rc('xtick')
    plt.rc('ytick')
    plt.errorbar(limb["week"],limb["comp_score"]["mean"],yerr=limb["comp_score"]["sem"] , uplims=True, lolims=True,label="Computational")
    plt.errorbar(limb["week"],limb["human_score"]["mean"],yerr=limb["human_score"]["sem"] , uplims=True, lolims=True,label="Human")
    plt.title( name+" Percent Slip Difference")
    plt.xlabel("Week")
    plt.ylabel("%slip")
    #plt.ylim(bottom=0)
    plt.legend()
    #invert x because preinjury is later alphabetically than postinjury
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/inflection_avg/perc_slip_"+name+'.png')

    plt.close()
    plt.figure()
    plt.rc('xtick')
    plt.rc('ytick')
    plt.errorbar(limb["week"],limb["comp_steps"]["mean"],yerr=limb["comp_steps"]["sem"] , uplims=True, lolims=True,label="Computational")
    plt.errorbar(limb["week"],limb["human_steps"]["mean"],yerr=limb["human_steps"]["sem"] , uplims=True, lolims=True,label="Human")
    plt.title( name+" Step Difference")
    plt.xlabel("Week")
    plt.ylabel("Number of Steps")
    #plt.ylim(bottom=0)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/inflection_avg/steps_"+name+'.png')

    plt.close()
    plt.figure()
    plt.rc('xtick')
    plt.rc('ytick')
    plt.errorbar(limb["week"],limb["comp_misses"]["mean"],yerr=limb["comp_misses"]["sem"] , uplims=True, lolims=True,label="Computational")
    plt.errorbar(limb["week"],limb["human_misses"]["mean"],yerr=limb["human_misses"]["sem"] , uplims=True, lolims=True,label="Human")
    plt.title( name+" Slip Difference")
    plt.xlabel("Week")
    plt.ylabel("Number of Slips")
    #plt.ylim(bottom=0)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/inflection_avg/slips_"+name+'.png')
print("All done")'''
