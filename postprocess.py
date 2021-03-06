import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.cluster import DBSCAN
from statistics import median



def func(x,a,b,c):
    '''
    A quadratic equation for use in curvefitting
    ...
    Parameters
    ----------
    x: 1d array. x values for the function
    a: number. quadratic coefficient
    b: number. linear coefficient
    c: number. constant

    Returns
    -------
    A quadradtic equation
    '''
    return a*x**2+b*x+c

def extract_limbs(df,network, limb):
    '''
    Separate the dataframe by limb
    ...
    Parameters
    ----------
    df: dataframe
    network: str. the first level of the multi-index.
    limb: str. the name of the column of the point in question as a string

    Returns
    -------
    A dataframe with just one point's x,y,and likelihood selected.
    The y coordinates have also had the curve deleted from it.
    '''
    df = df[network][limb]
    base  =  func(df['x'],*np.polyfit(rung_x,rung_y,2))
    df2 = pd.DataFrame()
    df2['x'] = df['x']
    df2["y"] = df["y"]-base
    df2['likelihood'] = df['likelihood']
    return df2

def likelihood_filter(df,threshold, fill=True):
    '''
    Parameters
    ----------
    df: dataframe
    threshold: number. the likelihood thresold that each point must be greater than or equal to
    fill: Boolean. determines if we fill NaNs in between. Default=True

    Returns
    -------
    A dataframe with points with confidence less than the specified threshold
    '''
    df.loc[df['likelihood']<=threshold] = np.nan
    df2 = df
    if fill == True:
        df2 = df2.ffill().add(df.bfill()).div(2)
    #df2 = df2.ffill()
    df2 = df2.dropna()
    df2 = df2.reset_index(drop=True)
    return df2

def not_outliers(data):
    '''
    Parameters
    ----------
    data: 1d array

    Returns
    -------
    a numpy array of the indexes of data with a z score less than 2
    '''
    z = np.abs(stats.zscore(data))
    return np.where(z<2)

def visible_limb_x_velocity_peaks(df,height,distance,direction):
    '''
    Parameters
    ----------
    df: dataframe
    height: number or nd array. from scipy.signal.find_peaks
    distance: number. from scipy.signal.find_peaks
    direction: str. "R" or "L" for the direction that the rat is going

    Returns
    -------
    List of indexes of peaks in the "forward" and "backward" direction
    '''
    if direction.upper() == "R":
        forward_x = find_peaks((np.gradient(df['x'])),height=height,distance=distance,prominence=1)
        backward_x = find_peaks(-1*np.gradient(df['x']),height=height,distance=distance)
    elif direction.upper()=="L":
        forward_x = find_peaks(-1*((np.gradient(df['x']))),height=height,distance=distance,prominence=1)
        backward_x = find_peaks(np.gradient(df['x']),height=height,distance=distance)
    return forward_x[0], backward_x[0]

def visible_limb_y_velocity_peaks(df,height,distance):
    '''
    Parameters
    ----------
    df: dataframe
    height: number or nd array
    distance: number

    Returns
    -------
    List of indexes that go upward and downward
    '''
    up_y = find_peaks(-1*np.gradient(df['y']),height=height,distance=distance)
    down_y = find_peaks(np.gradient(df['y']),height=height,distance=distance)
    return up_y[0],down_y[0]

def peak_list_union(list1,list2):
    '''
    Parameters
    ----------
    list1: list. first list that we want to join
    list2: list. second list that we want to join

    Returns
    -------
    List that is the union of the two input lists.
    '''
    lst = list(set(list1) | set(list2))
    return lst

def remove_close(lst):
    '''
    Currently unused
    ...
    Parameters
    ----------
    lst: list. List of indexes

    Returns
    -------
    list that has had elements that are less than 2 apart removed.
    '''
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
    '''
    Finds where the velocity in the x or y direction is close to 0. Currently not used
    ....
    Parameters
    ----------
    df: dataframe
    coord: str. "x" or "y"
    top_r: positive number. maximum allowed positive velocity
    bottom_r: positive number. maximum allowed negative velocity

    Returns
    -------
    Array of dataframe index values
    '''
    a = np.gradient(df[coord])
    return df.loc[np.where(np.logical_and(a<top_r,a>-bottom_r))].index

def zero_velocity_y_position():
    '''
    Finds the y coordinate where x and y velocity are close to 0. Used as a position threshold for peak-finding (for slips)

    Returns
    -------
    y value for the front limb and y value for the hind limb
    '''
    all_range = 0.3

    v_x_zero_fingers = zero_velocity_index(df_fingers,'x',all_range,all_range)
    v_y_zero_fingers = zero_velocity_index(df_fingers,'y',all_range,all_range)

    v_x_zero_toes = zero_velocity_index(df_toes,'x',all_range,all_range)
    v_y_zero_toes = zero_velocity_index(df_toes,'y',all_range,all_range)

    v_zero_front = list((set(v_x_zero_fingers)|set(v_y_zero_fingers)))
    v_zero_back = list((set(v_x_zero_toes)|set(v_y_zero_toes)))

    y_pos_front = (df_fingers["y"][v_zero_front]).max()
    y_pos_back = (df_toes["y"][v_zero_back]).max()

    return y_pos_front, y_pos_back

def find_y_position_peaks(df,thresh,dist):
    '''
    Finds the peak of the y coordinate.

    Parameters
    ----------
    df : dataframe
    thresh: number. height as defined in the scipy.signal.find_peaks documentaion
    dist: number. distance as defined in the find_peaks documentation

    Returns
    -------
    ndarray. indices of peaks in the y that satisfy the conditions.
    '''
    peaks = find_peaks(df['y'],height = 3,distance=dist,prominence=1)
    return peaks[0]


def find_clusters(df):
    '''
    Finds clusters of points that meet certain conditions

    Parameters
    ----------
    df: dataframe

    Returns
    -------
    List of indices of the center of the cluster
    '''
    #average y value
    avg = df['y'].mean()-5
    #remove values that are less (physically above) the "average." This is to prevent the clustering algorithm from counting clusters of points when the rat rears
    df = df.loc[df['y']>avg]
    #do the DBSCAN (Density based spatial clustering of applications with noise). fit to the x and y coordinates
    db = DBSCAN(eps = 22, min_samples = 3).fit(df)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    unique_labels = set(labels)
    cluster_center = []
    #get only the center of the cluster
    for k in unique_labels:
        class_member_mask = (labels == k)
        if len(df[class_member_mask & core_samples_mask].index)>1:
            center = round(median(list(df[class_member_mask & core_samples_mask].index)),0)
            cluster_center.append(center)
        else:
            continue
    return cluster_center

def clusters_y(df):
    '''
    Find the y values of the center of clusters. Intended to be used to separate steps from slips

    Parameters
    ----------
    df: dataframe

    Returns
    -------
    List of y values of the center of cluster "below" 0 (which is the theoretical y value of the rungs)
    '''
    avg = df['y'].mean()-5
    df = df.loc[df['y']>avg]
    db = DBSCAN(eps = 20, min_samples = 3).fit(df)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    unique_labels = set(labels)
    cluster_y = []
    for k in unique_labels:
        class_member_mask = (labels == k)
        if len(df[class_member_mask & core_samples_mask].index)>1:
            center_y = median(list(df['y'][class_member_mask & core_samples_mask]))
            cluster_y.append(center_y)
        else:
            continue
    return [y for y in cluster_y if y >0]

def inflection(df):
    '''
    Finds points of inflection in the acceleration in the x direction, which correspond to the transition between putting the paw down and picking it up

    Parameters
    ----------
    df: dataframe

    Returns
    -------
    1darray of indicies of those peaks
    '''
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
    return peak_x
