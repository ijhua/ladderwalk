# read the file
rats = ["MC61","MC87","MC30","MC70","MC45","MC78"]
#define handedness of the rats
right_handed = ["MC45","MC61","MC78","MC87","MC30","MC70"]
left_handed = []

#location of h5 files for analysis
folders = []
for rat in rats:
    folders+=glob.glob("/home/ml/Documents/Not_TADSS_Videos/"+rat+"/cut/dlc_output_16-810/*.h5")

#set up dataframe for future
scores = []
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
    if len(name.split("_")[3])==8:
        crossing=name.split("_")[3][-1]
    elif len(name.split("_")[3])>8:
        crossing = name.split("_")[3][-2:]
    #parameters for later
    likelihood_threshold = 0.1
    xheight = 20
    xdist = 4
    yheight = 5
    ydist = 4
    zero_threshold = 5

    #incorporate rung information that matches the current file
    #set rung file path
    path = f.split(".")[0].split("/")[:-2]+["dlc_output_rungs"]
    rung_folder = os.path.join(*(path))
    rung_file= f.split(".")[0].split("/")[-1].split("_")[0]+"_"+f.split(".")[0].split("/")[-1].split("_")[1]+"_"+f.split(".")[0].split("/")[-1].split("_")[2]+"_"+f.split(".")[0].split("/")[-1].split("_")[3]+"_"+f.split(".")[0].split("/")[-1].split("_")[4]+"_"+f.split(".")[0].split("/")[-1].split("_")[5]
    #read the file with the rung data
    rung_df = pd.read_hdf("/"+rung_folder+"/"+rung_file+"_"+"DLC_resnet50_LadderWalkMar12shuffle1_350000.h5")
    rung_df = rung_df['DLC_resnet50_LadderWalkMar12shuffle1_350000']
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
        df_wrist = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left wrist")
        df_fingers = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left fingers")
        #back left
        df_ankle = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left ankle")
        df_toes = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left toes")

        #rung curve fit
        x = likelihood_filter(df['DLC_resnet101_LadderWalkFeb13shuffle1_1030000']["left fingers"],0.1,fill=False)['x']
        y = likelihood_filter(df['DLC_resnet101_LadderWalkFeb13shuffle1_1030000']["left fingers"],0.1,fill=False)['y']

        x2 = likelihood_filter(df['DLC_resnet101_LadderWalkFeb13shuffle1_1030000']["left toes"],0.1,fill=False)['x']
        y2 = likelihood_filter(df['DLC_resnet101_LadderWalkFeb13shuffle1_1030000']["left toes"],0.1,fill=False)['y']
        plt.close()
        plt.plot(rung_x,func(rung_x,*np.polyfit(rung_x,rung_y,2)),label='rung curve fit',color='k')
        plt.scatter(rung_x,rung_y,label='Rungs')
        plt.scatter(x,y,label="Forelimb location")
        plt.scatter(x2,y2,label="Hindlimb location")
        plt.title(subject + " " + date + " "+ run + "Rungs Curve Fit")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.savefig("/home/ml/Documents/methods_figures/rungs/"+subject+"_"+date+'_'+run+"_rung_fit.png")
        plt.close()

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
        df_wrist = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right wrist")
        df_fingers = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right fingers")
        #back right
        df_ankle = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right ankle")
        df_toes = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right toes")

        x = likelihood_filter(df['DLC_resnet101_LadderWalkFeb13shuffle1_1030000']["right fingers"],0.1,fill=False)['x']
        y = likelihood_filter(df['DLC_resnet101_LadderWalkFeb13shuffle1_1030000']["right fingers"],0.1,fill=False)['y']

        x2 = likelihood_filter(df['DLC_resnet101_LadderWalkFeb13shuffle1_1030000']["right toes"],0.1,fill=False)['x']
        y2 = likelihood_filter(df['DLC_resnet101_LadderWalkFeb13shuffle1_1030000']["right toes"],0.1,fill=False)['y']
        plt.close()
        plt.plot(rung_x,func(rung_x,*np.polyfit(rung_x,rung_y,2)),label='rung curve fit',color='k')
        plt.scatter(rung_x,rung_y,label='Rungs')
        plt.scatter(x,y,label="Forelimb location")
        plt.scatter(x2,y2,label="Hindlimb location")
        plt.title(subject + " " + date + " "+ run + " Rungs Curve Fit ")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.savefig("/home/ml/Documents/methods_figures/rungs/"+subject+"_"+date+'_'+run+"_rung_fit.png")



    #filter dataframes by likelihood
    #front
    df_wrist = likelihood_filter(df_wrist,likelihood_threshold)
    df_fingers = likelihood_filter(df_fingers,likelihood_threshold)

    #back
    df_ankle = likelihood_filter(df_ankle,likelihood_threshold)
    df_toes = likelihood_filter(df_toes,likelihood_threshold)



    #get the x velocity peaks using clusters
    wrist_forward_list = find_clusters(df_wrist)
    fingers_forward_list = find_clusters(df_fingers)

    ankle_forward_list = find_clusters(df_ankle)
    toes_forward_list = find_clusters(df_toes)

    #after curve fit with steps
    x = df_fingers['x']
    y = df_fingers['y']
    x2 = df_toes['x']
    y2 = df_toes['y']
    plt.close()
    plt.plot(x,y,label='Forelimb location',color='tab:orange',marker='o')
    plt.plot(x2,y2,label='Hindlimb location',color='tab:green',marker='o')
    plt.plot(x[fingers_forward_list],y[fingers_forward_list],color='gold',markersize=12,marker='o',lw=0,label='Forelimb Steps')
    plt.plot(x2[toes_forward_list],y2[toes_forward_list],color='limegreen',markersize=12,marker='o',lw=0,label='Hindlimb Steps')
    plt.title(subject + " " + date + " "+ run + " Position After Correction with Step Clusters")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig("/home/ml/Documents/methods_figures/clusters/"+subject+"_"+date+'_'+run+"_position.png")
    plt.close()

    #join the two points on the front left limb
    front_forward = fingers_forward_list
    #front_forward = remove_close(sorted(front_forward))

    #join two points on the back left limb
    back_forward = toes_forward_list
    #back_forward = remove_close(sorted(back_forward))

    #figure out the y position threshold. it'll be when vx and vy are approximately 0
    y_pos_threshold_front,y_pos_threshold_back = zero_velocity_y_position()

    fingers_slip_list = find_y_position_peaks(df_fingers,y_pos_threshold_front,ydist)
    wrist_slip_list = find_y_position_peaks(df_wrist,y_pos_threshold_front,ydist)

    ankle_slip_list = find_y_position_peaks(df_ankle,y_pos_threshold_back,ydist)
    toes_slip_list = find_y_position_peaks(df_toes,y_pos_threshold_back,ydist)


    #join front lists of slips between the two points on the forelimb
    front_slip = peak_list_union(wrist_slip_list,fingers_slip_list)
    #join back lists for the two points on the hindlimb
    back_slip = peak_list_union(ankle_slip_list,toes_slip_list)

    #count the number of steps, hits and slips for left and right crossings
    if run[0] == "L":
        #number of x peaks is the number of total steps. I don't think there are really ever any backward peaks, so we'll just ignore them for now
        total_steps_fl = len(front_forward)#-len(fl_backward)
        total_steps_bl = len(back_forward)#-len(bl_backward)
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
        #number of x peaks is the number of total steps. I don't think there are really ever any backward peaks, so we'll just ignore them for now
        total_steps_fl = np.nan
        total_steps_bl = np.nan
        total_steps_fr = len(front_forward)#-len(fr_backward)
        total_steps_br = len(back_forward)#-len(br_backward)

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
    scores.append(score_front_l)
    scores.append(score_back_l)
    scores.append(score_front_r)
    scores.append(score_back_r)

#make the list into a dataframe with all of the scores
score_df = pd.DataFrame(scores,columns=score_cols)
#make the date into a datetime format
score_df["date"] = pd.to_datetime(score_df["date"])

#read the human_scores file
test_human = pd.read_csv("/home/ml/Documents/updated_human_scores.csv")
#test_human = test_human.ffill(axis=0)
#date column into datetime format
test_human['date'] = pd.to_datetime(test_human['date'])

#merge the computational and human score dataframes
all_score = score_df.merge(test_human,on=["subject","date","run","limb"])
#drop the empty rows (should just be the rows with the side of the rat that is further away)
all_score = all_score.dropna(axis=0)

all_score.to_csv("/home/ml/Documents/comparison_scores_6_mc_rats_clustering.csv")
print("Score calculation and comparison done.... Starting on graphs")

calcs=[]
#interate through every row in the dataframe.
#this way is slower than just subtracting columns, but it allows us to set the date of injury as 0
#TODO: make this section more efficient with time and maybe just make it a dict or datafraem for when there are more rats to deal with
for index,row in all_score.iterrows():
    #get the subject ID
    subject = row['subject']
    #Set the date of injury for each rat
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
    week_num = (row['date'] - date1).days/7
    #change week number into binary categories: pre and post injury
    if week_num <=0:
        week = "Preinjury"
    if week_num>0:
        week="Postinjury"
    #definethe limb column
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
#round the number of weeks (more useful when there are more than 2 relevant weeks of data)
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
    plt.savefig("/home/ml/Documents/methods_figures/histograms/steps_"+name+'.png')

    #difference in misses
    plt.close()
    plt.hist(limb["miss_diff"],label='Multipoint Difference')
    plt.legend()
    plt.title("Difference in number of misses "+name)
    plt.xlabel("Computation - Human")
    plt.ylabel("Number of Runs")
    plt.savefig("/home/ml/Documents/methods_figures/histograms/misses_"+name+'.png')


    #difference in hits
    plt.close()
    plt.hist(limb["hit_diff"],label='Multipoint Difference')
    plt.legend()
    plt.title("Difference in number of hits "+name)
    plt.xlabel("Computation - Human")
    plt.ylabel("Number of Runs")
    plt.savefig("/home/ml/Documents/methods_figures/histograms/hits_"+name+'.png')



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
    plt.ylim(bottom=0)
    plt.legend()
    #invert x because preinjury is later alphabetically than postinjury
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/average_comp/perc_slip_"+name+'.png')


    plt.close()
    plt.figure()
    plt.rc('xtick')
    plt.rc('ytick')
    plt.errorbar(limb["week"],limb["comp_steps"]["mean"],yerr=limb["comp_steps"]["sem"] , uplims=True, lolims=True,label="Computational")
    plt.errorbar(limb["week"],limb["human_steps"]["mean"],yerr=limb["human_steps"]["sem"] , uplims=True, lolims=True,label="Human")
    plt.title( name+" Step Difference")
    plt.xlabel("Week")
    plt.ylabel("Number of Steps")
    plt.ylim(bottom=0)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/average_comp/steps_"+name+'.png')

    plt.close()
    plt.figure()
    plt.rc('xtick')
    plt.rc('ytick')
    plt.errorbar(limb["week"],limb["comp_misses"]["mean"],yerr=limb["comp_misses"]["sem"] , uplims=True, lolims=True,label="Computational")
    plt.errorbar(limb["week"],limb["human_misses"]["mean"],yerr=limb["human_misses"]["sem"] , uplims=True, lolims=True,label="Human")
    plt.title( name+" Slip Difference")
    plt.xlabel("Week")
    plt.ylabel("Number of Slips")
    plt.ylim(bottom=0)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/average_comp/slips_"+name+'.png')
print("All done")
