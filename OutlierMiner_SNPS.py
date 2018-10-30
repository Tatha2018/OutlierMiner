# Comments

# 1. The same bayesian network as in Fig.5 is considered with an additional node E whose CPT is filled by Tatha

# 2. Input from user will be defined BN, minconf, maxconf, top_n and dataset

# 3. Only issue with the code is that the CPT need to be arranged properly to encounter a most probable bug in pgmpy library
#    that cannot properly assign the CPDs in the CPT and therefore will calculate wrong CPDs for the child nodes for one instance

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import pandas as pd
import numpy as np
import math

def OutlierMiner(Tatha_model,minconf,maxconf,top_n,data):
    # Create dataframe for holding data for JPDs
    df = pd.DataFrame(index = ['Parent', 'minsupp', 'CPD' , 'CPD_New'])
    
    # Calculate minsupp for all parent nodes
    for n in Tatha_model.nodes():
        test_str = Tatha_model.local_independencies(n).get_assertions()[0]
        target, independents, parents = [list(x) for x in test_str.get_assertion()]
        if parents == []:
            df.loc['Parent',n] = 1 # 1 for Root parent node (for reference)
        else:
            df.loc['Parent',n] = 2 # 2 for intermediate node (for reference)
            for ii in range(len(parents)):
                df.loc['minsupp',parents[ii]] = np.amin(Tatha_model.get_cpds(parents[ii]).values)
    for n in Tatha_model.nodes():
        if math.isnan(df.loc['minsupp',n]):
            df.loc['Parent',n] = 3 # for child node (for reference)
            
            
    jPD = pd.DataFrame(columns =['JPD'])
    for idx, instance in data.iterrows():
        # Calculate CPS of all child nodes
        for n in Tatha_model.nodes():
            test_str = Tatha_model.local_independencies(n).get_assertions()[0]
            target, independents, parents = [list(x) for x in test_str.get_assertion()]
            if parents == []:
                df.loc['CPD',n] = Tatha_model.get_cpds(target[0]).values[instance.loc[target]][0]
            else:
                temp_sol = []
                temp_sol.append([instance.loc[n]])
                for ii in range(len(parents)):
                    temp_sol.append([instance.loc[parents[ii]]])
                df.loc['CPD',n] = Tatha_model.get_cpds(n).values[temp_sol]
        # Update CPDs by using the R1 and R2 rule
        for n in Tatha_model.nodes():
            test_str = Tatha_model.local_independencies(n).get_assertions()[0]
            target, independents, parents = [list(x) for x in test_str.get_assertion()]
            if parents == []:
                df.loc['CPD_New',n] = df.loc['CPD',n]
            else:
                condition = 0
                for ii in range(len(parents)):
                    if ((df.loc['CPD',parents[ii]] == df.loc['minsupp',parents[ii]] and np.float64(df.loc['CPD',target]) >= maxconf) or\
                        (df.loc['CPD',parents[ii]] > df.loc['minsupp',parents[ii]] and np.float64(df.loc['CPD',target]) <= minconf)):
                        condition = condition + 1
                if condition > 0:
                    df.loc['CPD_New',n] = df.loc['CPD',n]
                elif condition == 0:
                    df.loc['CPD_New',n] = 1
        jpd = 1
        for n in df.loc['CPD_New',:]:
            jpd = jpd * n
        jPD.loc[idx,'JPD'] = jpd
#        print(df)
#        print(jpd)
    jPD = jPD.sort_values('JPD')
#    print(jPD)
    jPD_Sorted = jPD.sort_values('JPD',ascending = True).head(top_n)
    print(jPD_Sorted)
    return jPD_Sorted

# Data Set (by User)
data = pd.read_csv('TathaData.csv')

# Define thresholds (by User)
minconf = 0.1; maxconf = 0.7
top_n = 10

# Create the Bayesian network  (by User) [[[Can also be created from user given data set]]]
Tatha_model = BayesianModel([('A', 'C'), 
                              ('A', 'D'),
                              ('B', 'C'),
                              ('C', 'E'),
                              ('D', 'E')])

# Now defining the parameters and storing the CPT
cpd_A = TabularCPD(variable='A', variable_card=2,
                      values=[[0.4], [0.6]])
cpd_B = TabularCPD(variable='B', variable_card=2,
                       values=[[0.02], [0.98]])
cpd_D = TabularCPD(variable='D', variable_card=2,
                        values=[[0.25, 0.45],
                                [0.75, 0.55]],
                        evidence=['A'],
                        evidence_card=[2])
cpd_C = TabularCPD(variable='C', variable_card=2,
                      values=[[0.15, 0.55, 0.92, 0.4],
                              [0.85, 0.45, 0.08, 0.6]],
                      evidence=['A', 'B'], evidence_card=[2,2])
cpd_E = TabularCPD(variable='E', variable_card=2,
                      values=[[0.25, 0.65, 0.90, 0.6],
                              [0.75, 0.35, 0.1, 0.4]],
                      evidence=['C', 'D'], evidence_card=[2,2])

# Associating the parameters with the model structure.
Tatha_model.add_cpds(cpd_A, cpd_B, cpd_C, cpd_D, cpd_E)
# Calling the OutlierMiner function
OutlierMiner(Tatha_model,minconf,maxconf,top_n,data)

