import pandas as pd
from scipy.stats import ttest_ind

score1 = pd.read_csv("")
score2 = pd.read_csv("")

all = score1.merge(score2, on = ["subject","date","run","limb"])

df = pd.DataFrame()

df[["subject","date","run","limb"]] = all[["subject","date","run","limb"]]
df['date'] = pd.to_datetime(df['date'])


df['avg_step'] = all[["human_step","human_step.1"]].mean()
df['avg_hit'] = all[["human_hit","human_hit.1"]].mean()
df['avg_miss'] = all[["human_miss","human_miss.1"]].mean()

df.to_csv("/home/ml/Documents/trends/combined_scores.csv")

calcs = []

for index,row in all.iterrows():
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
    #define the main calculations between columns
    steps = row["avg_steps"]
    hits = row['avg_hits']
    miss = row["avg_misses"]
    score = row["avg_miss"]/row["avg_steps"]*100

    #append to the list
    calcs.append([subject,week,limb,steps,slips,hits,score])

calc = pd.DataFrame(calcs, columns=["subject","week","limb","steps","hits","miss","score"])

avg = calc.groupby(['week','limb'])['steps','hits','miss','score'].agg(["mean",'sem'])

#separate dataframe by limb
fd = avg.loc[avg["limb"] == "Dominant Front"]

fn = avg.loc[avg["limb"] =="Nondominant Front"]

bd = avg.loc[avg["limb"] =="Dominant Back"]

bn = avg.loc[avg["limb"] =="Nondominant Back"]

#list of limbs
limbs = [fd,fn,bd,bn]

t_p = []
#ttest
for limb in limbs:
    limb = limb.reset_index()
    name = limb["limb"][0]

    pre = limb.loc[limb["week"]=="Preinjury"]
    post = limb.loc[limb["week"]=="Postinjury"]
    for factor in ['steps','hits','miss','score']:
        #t-test
        a = pre[factor]
        b = post[factor]
        t, p = ttest_ind(a,b)
        print(name + "t = " + str(t))
        print(name + "p = " + str(p))
        t_p.append([name,factor,t,p])
tp_df = pd.DataFrame(t_p,columns=["limb","measurement",'t','p'])

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
    plt.savefig("/home/ml/Documents/trends/perc_slip_"+name+'.png')


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
    plt.savefig("/home/ml/Documents/trends/steps_"+name+'.png')

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
    plt.savefig("/home/ml/Documents/trends/slips_"+name+'.png')
print("All done")
