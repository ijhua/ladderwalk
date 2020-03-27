#open and separate out right or left limbs from multiindex dataframes
def func(x,a,b,c):
    return a*x**2+b*x+c

def extract_limbs(df,network, limb):
    df = df[network][limb]
    base  =  func(df['x'],*np.polyfit(rung_x,rung_y,2))
    df2 = pd.DataFrame()
    df2['x'] = df['x']
    df2["y"] = df["y"]-base
    df2['likelihood'] = df['likelihood']
    return df2

def likelihood_filter(df,threshold):
    df.loc[df['likelihood']<=threshold] = np.nan
    df2 = df
    df2 = df2.ffill().add(df.bfill()).div(2)
    #df2 = df2.ffill()
    df2 = df2.dropna()
    df2 = df2.reset_index(drop=True)
    return df2

def not_outliers(data):
    z = np.abs(stats.zscore(data))
    return np.where(z<1.8)

def visible_limb_x_velocity_peaks(df,height,distance,direction):
    if direction.upper() == "R":
        forward_x = find_peaks((np.gradient(df['x'])),height=height,distance=distance,prominence=1)#argrelmin(np.gradient(df['x']),mode='wrap')
        backward_x = find_peaks(-1*np.gradient(df['x']),height=height,distance=distance)
    elif direction.upper()=="L":
        forward_x = find_peaks(-1*((np.gradient(df['x']))),height=height,distance=distance,prominence=1)# argrelmin(-1*np.gradient(df['x']),mode='wrap')
        backward_x = find_peaks(np.gradient(df['x']),height=height,distance=distance)
    return forward_x[0], backward_x[0]

def visible_limb_y_velocity_peaks(df,height,distance):
    up_y = find_peaks(-1*np.gradient(df['y']),height=height,distance=distance)
    down_y = find_peaks(np.gradient(df['y']),height=height,distance=distance)
    return up_y[0],down_y[0]

def peak_list_union(list1,list2):
    lst = list(set(list1) | set(list2))
    return lst

def remove_close(lst):
    new_lst = []
    lst = sorted(lst)
    if len(lst)>1:
        for i in range(len(lst)-1):
            diff = abs(lst[i] - lst[i+1])
            if diff >2:
               new_lst.append(lst[i])
    elif len(lst)==1:
        new_lst = lst
    return new_lst

def zero_velocity_index(df,coord,top_r,bottom_r):
    #give the dataframe where the coord velocity is close to 0
    a = np.gradient(df[coord])
    return df.loc[np.where(np.logical_and(a<top_r,a>-bottom_r))].index

def zero_velocity_y_position():
    all_range = 0.4

    v_x_zero_fingers = zero_velocity_index(df_fingers,'x',all_range,all_range)
    v_y_zero_fingers = zero_velocity_index(df_fingers,'y',all_range,all_range)

    v_x_zero_toes = zero_velocity_index(df_toes,'x',all_range,all_range)
    v_y_zero_toes = zero_velocity_index(df_toes,'y',all_range,all_range)

    v_zero_front = list((set(v_x_zero_fingers)|set(v_y_zero_fingers)))
    v_zero_back = list((set(v_x_zero_toes)|set(v_y_zero_toes)))

    y_pos_front = (df_fingers["y"][v_zero_front]).max()+13
    y_pos_back = (df_toes["y"][v_zero_back]).max()+13

    return y_pos_front, y_pos_back

def find_y_position_peaks(front,back,dist):
    front_peaks = find_peaks(df_fingers['y'],height = front,distance=dist,prominence=1)
    back_peaks = find_peaks(df_toes['y'],height = back,distance=dist,prominence=1)
    return front_peaks[0],back_peaks[0]


def find_clusters(df):
    db = DBSCAN(eps = 25, min_samples = 3).fit(df)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    unique_labels = set(labels)
    cluster_center = []
    for k in unique_labels:
        class_member_mask = (labels == k)
        if len(df[class_member_mask & core_samples_mask].index)>1:
            center = round(median(list(df[class_member_mask & core_samples_mask].index)),0)
            cluster_center.append(center)
        else:
            continue
    return cluster_center
