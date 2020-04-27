# read the file
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
    if week_num <=0:
        week = "Preinjury"
    if week_num>0:
        week="Postinjury"


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

    #estimate distance
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

    rung_dists = np.array([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12])
    rung_dists = rung_dists[not_outliers(rung_dists)[0]]
    rung_median = median(rung_dists)


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
        #df_wrist = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"left wrist")
        df_fingers = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"left fingers")
        #back left
        #df_ankle = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"left ankle")
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
        #df_wrist = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"right wrist")
        df_fingers = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"right fingers")
        #back right
        #df_ankle = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"right ankle")
        df_toes = extract_limbs(df,'DLC_resnet50_LadderWalkFeb13shuffle1_450000',"right toes")

    #filter dataframes by likelihood
    #front
    #df_wrist = likelihood_filter(df_wrist,likelihood_threshold)
    df_fingers = likelihood_filter(df_fingers,likelihood_threshold)
    df_fingers.name = limb_front

    #back
    #df_ankle = likelihood_filter(df_ankle,likelihood_threshold)
    df_toes = likelihood_filter(df_toes,likelihood_threshold)
    df_toes.name = limb_back
    dfs = [df_fingers,df_toes]
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
    print("Graphs done")
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
                #distance between 2 adjacent rungs = 0.7cm
                #conversion: 1 pixel = 0.7/median
                dx = abs(df3['x'][0] - df3['x'][len(df3)-1])*(0.7/rung_median)
                dy = abs(df3['y'][0] - df3['y'][len(df3)-1])*(0.7/rung_median)
                dt = len(df3)*(1/framerate)
                vx = dx/dt
                vy = dy/dt
                '''all_t.append(dt)
                dist_x.append(dx)
                dist_y.append(dy)
                vels_x.append(vx)
                vels_y.append(vy)'''
                d = hypot(dx,dy)
                v = hypot(vx,vy)
                ind.append([subject,date,week,run,limb,framerate,col,dt,dx,dy,d,vx,vy,v])
            '''avg_t = mean(all_t)

            avg_d_x = mean(dist_x)

            avg_d_y = mean(dist_y)
            avg_d = hypot(avg_d_x,avg_d_y)
            avg_v_x = mean(vels_x)
            avg_v_y = mean(vels_y)
            avg_v = hypot(avg_v_x,avg_v_y)'''

            '''if len(new_df.columns.levels[0])>1:
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
                std_v_y = np.nan'''
            #data.append([subject,date,run,limb,framerate,num_windows,avg_t,std_t,avg_d_x,std_d_x,avg_d_y,std_d_y,avg_d,avg_v_x,std_v_x,avg_v_y,std_v_y,avg_v])
    count+=1
#data_df = pd.DataFrame(data,columns = ["subject","date","run","limb","framerate",'number of windows',"avg t (s)", "std t","avg dx (cm)","std dx","avg dy (cm)","std dy","avg d (cm)",'avg vx (cm/s)','std vx','avg vy (cm/s)',"std vy","avg v (cm/s)"])
#data_df["date"] = pd.to_datetime(data_df["date"])
#data_df.to_csv("/home/ml/Documents/step_stats.csv")

ind_df = pd.DataFrame(ind,columns=["subject","date","week","run","limb","framerate","step_no","time","dx","dy","distance","vx","vy","velocity"])
ind_df["date"] = pd.to_datetime(ind_df["date"])
ind_df.to_csv("/home/ml/Documents/individual_steps.csv")

calcs = []
for index,row in ind_df.iterrows():
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

    t = row["time"]
    dx = row["dx"]
    d = row["distance"]
    vx = row["vx"]
    v = row["velocity"]

    calcs.append([subject,week,limb,t,dx,d,vx,v])

calc_df = pd.DataFrame(calcs,columns=["subject","week","limb","t","dx","d","vx","v"])

#t tests
comb_df = calc_df
comb_df = comb_df.reset_index(drop=True)

comb_fd = comb_df.loc[comb_df["limb"]=="Dominant Front"]
comb_bd = comb_df.loc[comb_df["limb"]=="Dominant Back"]
comb_fn = comb_df.loc[comb_df["limb"]=="Nondominant Front"]
comb_bn = comb_df.loc[comb_df["limb"]=="Nondominant Back"]

ts_list = []

t_fd = stats.ttest_ind(comb_fd.loc[comb_fd["week"]=="Preinjury"].iloc[:,3],comb_fd.loc[comb_fd["week"]=="Postinjury"].iloc[:,3])[1]
dx_fd = stats.ttest_ind(comb_fd.loc[comb_fd["week"]=="Preinjury"].iloc[:,4],comb_fd.loc[comb_fd["week"]=="Postinjury"].iloc[:,4])[1]
d_fd = stats.ttest_ind(comb_fd.loc[comb_fd["week"]=="Preinjury"].iloc[:,5],comb_fd.loc[comb_fd["week"]=="Postinjury"].iloc[:,5])[1]
v_fd = stats.ttest_ind(comb_fd.loc[comb_fd["week"]=="Preinjury"].iloc[:,6],comb_fd.loc[comb_fd["week"]=="Postinjury"].iloc[:,6])[1]
vx_fd = stats.ttest_ind(comb_fd.loc[comb_fd["week"]=="Preinjury"].iloc[:,7],comb_fd.loc[comb_fd["week"]=="Postinjury"].iloc[:,7])[1]

ts_list.append(["Dominant Front", t_fd,dx_fd,d_fd,v_fd,vx_fd])

t_bd = stats.ttest_ind(comb_bd.loc[comb_bd["week"]=="Preinjury"].iloc[:,3],comb_bd.loc[comb_bd["week"]=="Postinjury"].iloc[:,3])[1]
dx_bd = stats.ttest_ind(comb_bd.loc[comb_bd["week"]=="Preinjury"].iloc[:,4],comb_bd.loc[comb_bd["week"]=="Postinjury"].iloc[:,4])[1]
d_bd = stats.ttest_ind(comb_bd.loc[comb_bd["week"]=="Preinjury"].iloc[:,5],comb_bd.loc[comb_bd["week"]=="Postinjury"].iloc[:,5])[1]
v_bd = stats.ttest_ind(comb_bd.loc[comb_bd["week"]=="Preinjury"].iloc[:,6],comb_bd.loc[comb_bd["week"]=="Postinjury"].iloc[:,6])[1]
vx_bd = stats.ttest_ind(comb_bd.loc[comb_bd["week"]=="Preinjury"].iloc[:,7],comb_bd.loc[comb_bd["week"]=="Postinjury"].iloc[:,7])[1]

ts_list.append(["Dominant Back", t_bd,dx_bd,d_bd,v_bd,vx_bd])


t_fn = stats.ttest_ind(comb_fn.loc[comb_fn["week"]=="Preinjury"].iloc[:,3],comb_fn.loc[comb_fn["week"]=="Postinjury"].iloc[:,3])[1]
dx_fn = stats.ttest_ind(comb_fn.loc[comb_fn["week"]=="Preinjury"].iloc[:,4],comb_fn.loc[comb_fn["week"]=="Postinjury"].iloc[:,4])[1]
d_fn = stats.ttest_ind(comb_fn.loc[comb_fn["week"]=="Preinjury"].iloc[:,5],comb_fn.loc[comb_fn["week"]=="Postinjury"].iloc[:,5])[1]
v_fn = stats.ttest_ind(comb_fn.loc[comb_fn["week"]=="Preinjury"].iloc[:,6],comb_fn.loc[comb_fn["week"]=="Postinjury"].iloc[:,6])[1]
vx_fn = stats.ttest_ind(comb_fn.loc[comb_fn["week"]=="Preinjury"].iloc[:,7],comb_fn.loc[comb_fn["week"]=="Postinjury"].iloc[:,7])[1]

ts_list.append(["Nondominant Front", t_fn,dx_fn,d_fn,v_fn,vx_fn])


t_bn = stats.ttest_ind(comb_bn.loc[comb_bn["week"]=="Preinjury"].iloc[:,3],comb_bn.loc[comb_bn["week"]=="Postinjury"].iloc[:,3])[1]
dx_bn = stats.ttest_ind(comb_bn.loc[comb_bn["week"]=="Preinjury"].iloc[:,4],comb_bn.loc[comb_bn["week"]=="Postinjury"].iloc[:,4])[1]
d_bn = stats.ttest_ind(comb_bn.loc[comb_bn["week"]=="Preinjury"].iloc[:,5],comb_bn.loc[comb_bn["week"]=="Postinjury"].iloc[:,5])[1]
v_bn = stats.ttest_ind(comb_bn.loc[comb_bn["week"]=="Preinjury"].iloc[:,6],comb_bn.loc[comb_bn["week"]=="Postinjury"].iloc[:,6])[1]
vx_bn = stats.ttest_ind(comb_bn.loc[comb_bn["week"]=="Preinjury"].iloc[:,7],comb_bn.loc[comb_bn["week"]=="Postinjury"].iloc[:,7])[1]

ts_list.append(["Nondominant Back", t_bn,dx_bn,d_bn,vx_bn,v_bn])

t_df = pd.DataFrame(ts_list, columns=["limb","t_t","t_dx","t_d","t_vx","t_v"])

#trends (calculate mean and sem)
trend = calc_df.groupby(["week","limb"])["t","dx","d","vx","v"].agg(["mean","sem"])
trend = trend.reset_index()
trends = trend.merge(t_df,on=['limb'])
trends.to_csv("/home/ml/Documents/avg_of_all_steps.csv")

fd = trends.loc[trends["limb"]=="Dominant Front"]
bd = trends.loc[trends["limb"]=="Dominant Back"]
fn = trends.loc[trends["limb"]=="Nondominant Front"]
bn = trends.loc[trends["limb"]=="Nondominant Back"]

limbs = [fd,bd,fn,bn]
for limb in limbs:
    limb = limb.reset_index()
    name = limb["limb"][0]

    plt.close()
    '''if limb['t_t'][0]<0.05:
        plt.scatter("Postinjury",limb[('t', 'mean')].max(),marker="$*$")
    elif limb['t_t'][0]<0.001:
        plt.scatter("Postinjury",limb[('t', 'mean')].max(),marker="$**$")
    plt.annotate(str(round(limb['t_t'][0],5)),("Postinjury",limb[('t', 'mean')].max()))'''
    plt.errorbar(limb[('week', '')],limb[('t', 'mean')],yerr=limb[('t', 'sem')],uplims=True, lolims=True)
    plt.xlabel("Week")
    plt.ylabel("Time (sec)")
    plt.title(name + " Avg Time Per Step")
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/loc_trends/"+name+"_avg_time.png")

    plt.close()
    '''if limb['t_d'][0]<0.05:
        plt.scatter("Postinjury",limb[('d', 'mean')].max(),marker="$*$")
    elif limb['t_d'][0]<0.001:
        plt.scatter("Postinjury",limb[('d', 'mean')].max(),marker="$**$")
    plt.annotate(str(round(limb['t_d'][0],5)),("Postinjury",limb[('d', 'mean')].max()))'''
    plt.errorbar(limb[('week', '')],limb['d', 'mean'],yerr=limb[('d', 'sem')],uplims=True, lolims=True)
    plt.xlabel("Week")
    plt.ylabel("Distance (cm)")
    plt.title(name + " Avg Step Distance")
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/loc_trends/"+name+"_avg_distance.png")

    plt.close()
    '''if limb['t_dx'][0]<0.05:
        plt.scatter("Postinjury",limb[('dx', 'mean')].max(),marker="$*$")
    elif limb['t_dx'][0]<0.001:
        plt.scatter("Postinjury",limb[('dx', 'mean')].max(),marker="$**$")
    plt.annotate(str(round(limb['t_dx'][0],5)),("Postinjury",limb[('dx', 'mean')].max()))'''
    plt.errorbar(limb[('week', '')],limb[('dx', 'mean')],yerr=limb[('dx', 'sem')],uplims=True, lolims=True)
    plt.xlabel("Week")
    plt.ylabel("Distance (cm)")
    plt.title(name + " Avg x Component of Step Distance")
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/loc_trends/"+name+"_avg_x_distance.png")

    plt.close()
    '''if limb['t_v'][0]<0.05:
        plt.scatter("Postinjury",limb[('v', 'mean')].max(),marker="$*$")
    elif limb['t_v'][0]<0.001:
        plt.scatter("Postinjury",limb[('v', 'mean')].max(),marker="$**$")'''
    plt.annotate(str(round(limb['t_v'][0],5)),("Postinjury",limb[('v', 'mean')].max()))
    plt.errorbar(limb[('week', '')],limb[('v', 'mean')],yerr=limb[('v', 'sem')],uplims=True, lolims=True)
    plt.xlabel("Week")
    plt.ylabel("Velocity (cm/sec)")
    plt.title(name + " Avg Step Velocity")
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/loc_trends/"+name+"_avg_velocity.png")

    plt.close()
    '''if limb['t_vx'][0]<0.05:
        plt.scatter("Postinjury",limb[('vx', 'mean')].max(),marker="$*$")
    elif limb['t_vx'][0]<0.001:
        plt.scatter("Postinjury",limb[('vx', 'mean')].maxs(),marker="$**$")'''
    plt.annotate(str(round(limb['t_vx'][0],5)),("Postinjury",limb[('vx', 'mean')].max()))
    plt.errorbar(limb[('week', '')],limb[('vx', 'mean')],yerr=limb[('vx', 'sem')],uplims=True, lolims=True)
    plt.xlabel("Week")
    plt.ylabel("Velocity (cms/sec)")
    plt.title(name + " Avg x Component of Step Velocity")
    plt.gca().invert_xaxis()
    plt.savefig("/home/ml/Documents/methods_figures/loc_trends/"+name+"_avg_x_velocity.png")

    plt.close()
print("Done")
