# read the file
rats = ["MC61","MC87","MC30","MC70","MC45","MC78"]
#define handedness of the rats
right_handed = ["MC45","MC61","MC78","MC87","MC30","MC70"]
left_handed = []

count = 1
#location of h5 files for analysis
folders = []
for rat in rats:
    folders+=glob.glob("/home/ml/Documents/Not_TADSS_Videos/"+rat+"/cut/dlc_output_16-810/*.h5")
data = []
#iterate through every file
for f in folders:
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


    video = cv2.VideoCapture(f.split('.')[0]+"_labeled.mp4")
    framerate = video.get(cv2.CAP_PROP_FPS)

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

    print("getting dfs")
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
        #df_wrist = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left wrist")
        df_fingers = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left fingers")
        #back left
        #df_ankle = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left ankle")
        df_toes = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left toes")

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
        #df_wrist = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right wrist")
        df_fingers = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right fingers")
        #back right
        #df_ankle = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right ankle")
        df_toes = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right toes")

    #filter dataframes by likelihood
    #front
    #df_wrist = likelihood_filter(df_wrist,likelihood_threshold)
    df_fingers = likelihood_filter(df_fingers,likelihood_threshold)
    df_fingers.name = limb_front

    #back
    #df_ankle = likelihood_filter(df_ankle,likelihood_threshold)
    df_toes = likelihood_filter(df_toes,likelihood_threshold)
    df_toes.name = limb_back
    '''    dfs = [df_fingers,df_toes]
        for df in dfs:
            limb = df.name
            second_x = np.gradient(np.gradient(df.x))
            pos_peak_x = find_peaks(second_x,prominence=5,width=1,height=10)[0]
            neg_peak_x = find_peaks(-second_x,prominence=5,width=1,height=10)[0]
            peak_x = np.append(pos_peak_x,neg_peak_x)
            peak_x = np.sort(peak_x)

            plt.close()
            plt.plot(df.x,second_x)
            plt.scatter(df.x.iloc[peak_x],second_x[peak_x])
            plt.xlabel('x')
            plt.ylabel('x acceleration')
            plt.title(subject + " " + date + " "+ run + " "+ limb + " Second Derivative Peaks")
            plt.savefig("/home/ml/Documents/methods_figures/inflection/"+subject+"_"+date+"_"+run+"_"+limb+"_"+"poi.png")

            plt.close()
            plt.plot(df.x,df.y)
            plt.scatter(df.x.iloc[peak_x],df.y.iloc[peak_x])
            plt.gca().invert_yaxis()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(subject + " " + date + " "+ run + " "+ limb + " Position Windows")
            plt.savefig("/home/ml/Documents/methods_figures/inflection/"+subject+"_"+date+"_"+run+"_"+limb+"_"+"window.png")
            plt.close()

    print("Done")'''
    dfs = [df_fingers,df_toes]
    for df in dfs:
        limb = df.name
        second_x = np.gradient(np.gradient(df.x))
        pos_peak_x = find_peaks(second_x,prominence=5,width=1,height=10)[0]
        neg_peak_x = find_peaks(-second_x,prominence=5,width=1,height=10)[0]
        peak_x = np.append(pos_peak_x,neg_peak_x)
        peak_x = np.sort(peak_x)

        new_index = pd.MultiIndex(levels=[[],[]],codes=[[],[]],names=["step","coords"])
        new_df = pd.DataFrame(columns=new_index)
        a=0
        for i,k in zip(peak_x[0::2], peak_x[1::2]):
            df2 = df.iloc[range(i,k+1)]
            df2 = df2.drop("likelihood",axis=1)
            df2.x = df2.x - df2.x.iloc[0]
            df2.y = df2.y - df2.y.iloc[0]
            df2 = df2.reset_index()
            if a>0:
                while len(df2)>len(new_df):
                    new_df = new_df.append([[]]).reset_index(drop=True)
            new_df['step_'+str(a),'x']= df2.x
            new_df['step_'+str(a),'y']= df2.y
            a+=1
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
                dx = abs(df3['x'][0] - df3['x'][len(df3)-1])
                dy = abs(df3['y'][0] - df3['y'][len(df3)-1])
                dt = len(df3)*(1/framerate)
                v_x = dx/dt
                v_y = dy/dt
                all_t.append(dt)
                dist_x.append(dx)
                dist_y.append(dy)
                vels_x.append(v_x)
                vels_y.append(v_y)
            avg_t = mean(all_t)

            avg_d_x = mean(dist_x)

            avg_d_y = mean(dist_y)
            avg_d = hypot(avg_d_x,avg_d_y)
            avg_v_x = mean(vels_x)
            avg_v_y = mean(vels_y)
            avg_v = hypot(avg_v_x,avg_v_y)

            if len(new_df.columns.levels[0])>1:
                std_t = stdev(all_t)
                std_d_x = stdev(dist_x)
                std_d_y = stdev(dist_y)
                std_v_x = stdev(vels_x)
                std_v_y = stdev(vels_y)
            else:
                std_t = np.nan
                std_d_x = np.nan
                std_d_y = np.nan
                std_v_x = np.nan
                std_v_y = np.nan
            data.append([subject,date,run,limb,framerate,num_windows,avg_t,std_t,avg_d_x,std_d_x,avg_d_y,std_d_y,avg_d,avg_v_x,std_v_x,avg_v_y,std_v_y,avg_v])
    count+=1
data_df = pd.DataFrame(data,columns = ["subject","date","run","limb","framerate",'number of windows',"avg t (s)", "std t","avg dx (px)","std dx","avg dy (px)","std dy","avg d (px)",'avg vx (px/s)','std vx','avg vy (px/s)',"std vy","avg v (px/s)"])
data_df["date"] = pd.to_datetime(data_df["date"])
data_df.to_csv("/home/ml/Documents/step_stats.csv")


print("Calculating dates")
calcs = []
for index,row in data_df.iterrows():
    #get the subject ID
    subject = row['subject']
    #Set the date of injury for each rat
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
    week_num = (row['date'] - date1).days/7
    #change week number into binary categories: pre and post injury
    if week_num <=0:
        week = "Preinjury"
    if week_num>0:
        week="Postinjury"
    #definethe limb column
    limb = row['limb']

    avg_t = row["avg t (s)"]
    avg_d_x = row["avg dx (px)"]
    avg_d = row["avg d (px)"]
    avg_v = row["avg v (px/s)"]
    avg_v_x = row["avg vx (px/s)"]

    calcs.append([subject,week,limb,avg_t,avg_d_x,avg_d,avg_v,avg_v_x])

print("Calculating Weekly Averages")
calc_df = pd.DataFrame(calcs,columns=["subject","week","limb","avg t (s)","avg dx (px)","avg d (px)","avg v (px/s)","avg vx (px/s)"])
#trends
trends = calc_df.groupby(["week","limb"])["avg t (s)","avg dx (px)","avg d (px)","avg v (px/s)","avg vx (px/s)"].agg(["mean","sem"])
trends = trends.reset_index()
trends.to_csv("/home/ml/Documents/trends.csv")

fd = trends.loc[trends["limb"]=="Dominant Front"]
bd = trends.loc[trends["limb"]=="Dominant Back"]
fn = trends.loc[trends["limb"]=="Nondominant Front"]
bn = trends.loc[trends["limb"]=="Nondominant Back"]

print("making graphs")
limbs = [fd,bd,fn,bn]
for limb in limbs:
    limb = limb.reset_index()
    name = limb["limb"][0]

    plt.close()
    plt.errorbar(limb["week"],limb["avg t (s)"]["mean"],yerr=limb["avg t (s)"]["sem"],uplims=True, lolims=True)
    plt.xlabel("Week")
    plt.ylabel("Time (sec)")
    plt.title(name + " Avg Time Per Step")
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/loc_trends/"+name+"_avg_time.png")

    plt.close()
    plt.errorbar(limb["week"],limb["avg d (px)"]["mean"],yerr=limb["avg d (px)"]["sem"],uplims=True, lolims=True)
    plt.xlabel("Week")
    plt.ylabel("Distance (pixels)")
    plt.title(name + " Avg Step Distance")
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/loc_trends/"+name+"_avg_distance.png")

    plt.close()
    plt.errorbar(limb["week"],limb["avg dx (px)"]["mean"],yerr=limb["avg dx (px)"]["sem"],uplims=True, lolims=True)
    plt.xlabel("Week")
    plt.ylabel("Distance (pixels)")
    plt.title(name + " Avg x Component of Step Distance")
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/loc_trends/"+name+"_avg_x_distance.png")

    plt.close()
    plt.errorbar(limb["week"],limb["avg v (px/s)"]["mean"],yerr=limb["avg v (px/s)"]["sem"],uplims=True, lolims=True)
    plt.xlabel("Week")
    plt.ylabel("Velocity (pixels/sec)")
    plt.title(name + " Avg Step Velocity")
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/loc_trends/"+name+"_avg_velocity.png")

    plt.close()
    plt.errorbar(limb["week"],limb["avg vx (px/s)"]["mean"],yerr=limb["avg vx (px/s)"]["sem"],uplims=True, lolims=True)
    plt.xlabel("Week")
    plt.ylabel("Velocity (pixels/sec)")
    plt.title(name + " Avg x Component of Step Velocity")
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/loc_trends/"+name+"_avg_x_velocity.png")

    plt.close()
