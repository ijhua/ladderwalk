# read the file
rats = ["MC61","MC78","MC87","MC30","MC70","MC45"]
right_handed = ["MC45","MC61","MC78","MC87","MC30","MC70"]
left_handed = []
folders = []
for rat in rats:
    folders+=glob.glob("/home/ml/Documents/Not_TADSS_Videos/"+rat+"/cut/dlc_output_16-810/*.h5")

scores = []
score_cols = ["subject", "date", "run", "crossing","limb","comp_hits","comp_misses","comp_steps"]
for f in folders:
    df = pd.read_hdf(f)
    name=f.split("/")[8]
    run = name.split("_")[2]
    subject = name.split("_")[0]
    date = name.split("_")[1]
    crossing=name.split("_")[3][-1]
    likelihood_threshold = 0.1
    xheight = 20
    xdist = 4
    yheight = 5
    ydist = 4
    zero_threshold = 5

    path = f.split(".")[0].split("/")[:-2]+["dlc_output_rungs"]
    rung_folder = os.path.join(*(path))
    rung_file= f.split(".")[0].split("/")[-1].split("_")[0]+"_"+f.split(".")[0].split("/")[-1].split("_")[1]+"_"+f.split(".")[0].split("/")[-1].split("_")[2]+"_"+f.split(".")[0].split("/")[-1].split("_")[3]+"_"+f.split(".")[0].split("/")[-1].split("_")[4]+"_"+f.split(".")[0].split("/")[-1].split("_")[5]
    rung_df = pd.read_hdf("/"+rung_folder+"/"+rung_file+"_"+"DLC_resnet50_LadderWalkMar12shuffle1_350000.h5")
    rung_df = rung_df['DLC_resnet50_LadderWalkMar12shuffle1_350000']
    rung_list = []
    for i in range(1,63):
        rung_list.append("rung_"+str(i))
    for rung in rung_list:
        rung_df[rung]=likelihood_filter(rung_df[rung],0.8)
    rung_mean = rung_df.agg(["mean","sem"])
    rung_x = np.empty(shape=(63,1))
    rung_y = np.empty(shape=(63,1))
    for rung in rung_list:
        num = int(rung.split("_")[-1])
        rung_x[num]=rung_mean[rung]["x"]["mean"]
        rung_y[num]=rung_mean[rung]["y"]["mean"]

    rung_x = rung_x[~np.isnan(rung_x)]
    rung_y = rung_y[~np.isnan(rung_y)]
    rung_x = rung_x[not_outliers(rung_y)]
    rung_y = rung_y[not_outliers(rung_y)]
    #popt, pcov = curve_fit(func, rung_x, rung_y)

    #left crossings
    if run[0] == "L":
        if subject in right_handed:
            limb_front = "Nondominant Front"
            limb_back = "Nondominant Back"
        elif subject in left_handed:
            limb_front = "Dominant Front"
            limb_back = "Dominant Back"
        #split all the limbs
        #frontleft
        df_wrist = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left wrist")
        df_fingers = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left fingers")
        #df_elbow = extract_limbs(df,"left elbow")
        #back left
        df_ankle = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left ankle")
        df_toes = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left toes")
    #right side
    if run[0] == "R":
        if subject in left_handed:
            limb_front = "Nondominant Front"
            limb_back = "Nondominant Back"
        elif subject in right_handed:
            limb_front = "Dominant Front"
            limb_back = "Dominant Back"
        #split all the limbs
        #frontright
        df_wrist = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right wrist")
        df_fingers = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right fingers")
        #df_elbow = extract_limbs(df,"right elbow")
        #back right
        df_ankle = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right ankle")
        df_toes = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right toes")
    #filter by likelihood
    #frontleft
    df_wrist = likelihood_filter(df_wrist,likelihood_threshold)
    df_fingers = likelihood_filter(df_fingers,likelihood_threshold)
    #df_elbow = likelihood_filter(df_elbow,likelihood_threshold)
    #back left
    df_ankle = likelihood_filter(df_ankle,likelihood_threshold)
    df_toes = likelihood_filter(df_toes,likelihood_threshold)

    #get the x velocity peaks
    wrist_forward_list = find_clusters(df_wrist)
    fingers_forward_list = find_clusters(df_fingers)

    ankle_forward_list = find_clusters(df_ankle)
    toes_forward_list = find_clusters(df_toes)

    #join the two points on the front left limb
    front_forward = fingers_forward_list
    #front_forward = remove_close(sorted(front_forward))

    #join two points on the back left limb
    back_forward = toes_forward_list
    #back_forward = remove_close(sorted(back_forward))

    #figure out the y position threshold. it'll be when vx and vy are approximately 0
    y_pos_threshold_front,y_pos_threshold_back = zero_velocity_y_position()

    y_pos_peaks_front,y_pos_peaks_back = find_y_position_peaks(y_pos_threshold_front,y_pos_threshold_back,ydist)


    #join front lists
    front_up = peak_list_union(wrist_up_list,fingers_up_list)
    #join back lists
    back_up = peak_list_union(ankle_up_list,toes_up_list)

    if run[0] == "L":
        #number of x peaks is the number of total steps. I don't think there are really ever any backward peaks, so we'll just ignore them for now
        total_steps_fl = len(front_forward)#-len(fl_backward)
        total_steps_bl = len(back_forward)#-len(bl_backward)
        total_steps_fr = np.nan
        total_steps_br = np.nan

        slip_count_fl = len(y_pos_peaks_front)
        slip_count_bl = len(y_pos_peaks_back)
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
        slip_count_fr = len(y_pos_peaks_front)
        slip_count_br = len(y_pos_peaks_back)

        hit_count_fl = np.nan
        hit_count_bl = np.nan
        hit_count_fr = total_steps_fr - slip_count_fr
        hit_count_br = total_steps_br - slip_count_br

    score_front_l = [subject,date,run,crossing,limb_front,hit_count_fl,slip_count_fl,total_steps_fl]
    score_back_l = [subject,date,run,crossing,limb_back,hit_count_bl,slip_count_bl,total_steps_bl]
    score_front_r = [subject,date,run,crossing,limb_front,hit_count_fr,slip_count_fr,total_steps_fr]
    score_back_r = [subject,date,run,crossing,limb_back,hit_count_br,slip_count_br,total_steps_br]

    scores.append(score_front_l)
    scores.append(score_back_l)
    scores.append(score_front_r)
    scores.append(score_back_r)

score_df = pd.DataFrame(scores,columns=score_cols)
score_df["date"] = pd.to_datetime(score_df["date"])

#human_scores
test_human = pd.read_csv("/home/ml/Documents/updated_human_scores.csv")
#test_human = test_human.ffill(axis=0)
test_human['date'] = pd.to_datetime(test_human['date'])

all_score = score_df.merge(test_human,on=["subject","date","run","limb"])
all_score = all_score.dropna()

all_score.to_csv("/home/ml/Documents/comparison_scores_6_mc_rats_clustering.csv")

calcs=[]

for index,row in all_score.iterrows():
    subject = row['subject']
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
    if week_num <=0:
        week = "Preinjury"
    if week_num>0:
        week="Postinjury"
    limb = row['limb']
    if row["comp_steps"] != 0:
        comp_score = row["comp_misses"]/row["comp_steps"]*100
    else:
        comp_score = np.nan
    comp_steps = row["comp_steps"]
    comp_slips = row["comp_misses"]
    comp_hits = row['comp_hits']
    human_score = row["human_miss"]/row["human_steps"]*100
    human_steps = row["human_steps"]
    human_miss = row["human_miss"]
    human_hits = row["human_hit"]
    calcs.append([subject,week,limb,comp_score,comp_steps,comp_slips,comp_hits,human_score,human_steps,human_miss,human_hits])
calc_df = pd.DataFrame(calcs,columns=["subject","week","limb","comp_score","comp_steps","comp_misses","comp_hits","human_score","human_steps","human_misses","human_hits"])
#calc_df = calc_df.round({"week":0})

#for non-averaged
new_calc = calc_df
new_calc["step_diff"] = new_calc["comp_steps"]-new_calc["human_steps"]
new_calc["miss_diff"] = new_calc["comp_misses"]-new_calc["human_misses"]
new_calc["hit_diff"] = new_calc["comp_hits"]-new_calc["human_hits"]

cfd = new_calc.loc[new_calc["limb"] == "Dominant Front"]

cfn = new_calc.loc[new_calc["limb"] =="Nondominant Front"]

cbd = new_calc.loc[new_calc["limb"] =="Dominant Back"]

cbn = new_calc.loc[new_calc["limb"] =="Nondominant Back"]

calc_limbs = [cfd,cfn,cbd,cbn]

for limb in calc_limbs:
    limb = limb.reset_index()
    name = limb["limb"][0]
    plt.close()
    plt.hist(limb["step_diff"],label='Multipoint Difference',alpha=0.5)
    plt.legend()
    plt.title("Difference in number of steps "+name)
    plt.xlabel("Computation - Human")
    plt.ylabel("Number of Runs")
    plt.show()

    plt.close()
    plt.hist(limb["miss_diff"],label='Multipoint Difference',alpha=0.5)
    plt.legend()
    plt.title("Difference in number of misses "+name)
    plt.xlabel("Computation - Human")
    plt.ylabel("Number of Runs")
    plt.show()

    plt.close()
    plt.hist(limb["hit_diff"],label='Multipoint Difference',alpha=0.5)
    plt.legend()
    plt.title("Difference in number of hits "+name)
    plt.xlabel("Computation - Human")
    plt.ylabel("Number of Runs")
    plt.show()

df_new = calc_df.groupby(["week","limb"])["comp_score","comp_steps","comp_misses","human_score","human_steps","human_misses"].agg(["mean",'sem'])

df_new=df_new.reset_index()
df_new = df_new.sort_values(by=["week"])

fd = df_new.loc[df_new["limb"] == "Dominant Front"]

fn = df_new.loc[df_new["limb"] =="Nondominant Front"]

bd = df_new.loc[df_new["limb"] =="Dominant Back"]

bn = df_new.loc[df_new["limb"] =="Nondominant Back"]

limbs = [fd,fn,bd,bn]
for limb in limbs:
    limb = limb.reset_index()
    name = limb["limb"][0]

    plt.close()
    plt.figure()
    plt.rc('xtick')
    plt.rc('ytick')
    plt.errorbar(limb["week"],limb["comp_score"]["mean"],yerr=limb["comp_score"]["sem"] , uplims=True, lolims=True,label="Computational")
    plt.errorbar(limb["week"],limb["human_score"]["mean"],yerr=limb["human_score"]["sem"] , uplims=True, lolims=True,label="Human")
    plt.title( name+" Slip Difference")
    plt.xlabel("Week")
    plt.ylabel("%slip")
    plt.ylim(bottom=0)
    plt.legend()
    plt.gca().invert_xaxis()
    #plt.savefig("/home/ml/Documents/"+name+"_weekly_number_score_MC_rats.png")
    plt.show()
    plt.figure()
    plt.rc('xtick')
    plt.rc('ytick')
    plt.errorbar(limb["week"],limb["comp_steps"]["mean"],yerr=limb["comp_steps"]["sem"] , uplims=True, lolims=True,label="Computational")
    plt.errorbar(limb["week"],limb["human_steps"]["mean"],yerr=limb["human_steps"]["sem"] , uplims=True, lolims=True,label="Human")
    plt.title( name+" Slip Difference")
    plt.xlabel("Week")
    plt.ylabel("Number of Steps")
    plt.ylim(bottom=0)
    plt.legend()
    plt.gca().invert_xaxis()
    #plt.savefig("/home/ml/Documents/"+name+"_weekly_step_count_MC_rats.png")
    plt.show()
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
    #plt.savefig("/home/ml/Documents/"+name+"_weekly_slip_count_MC_rats.png")
    plt.show()
