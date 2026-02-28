# Exploratory analysis of Medicare drug cost data 2011-2015
> "Health care systems world-wide are under pressure due to the high costs associated with disease. In this post, I performed an analysis of Medicare data in the USA. Furthermore I used a drug-disease open database to cluster the costs by disease. I identified the most expensive diseases (mostly chronic diseases such as Diabetes) and the most expensive medicines."

- toc:true
- branch: master
- badges: true
- comments: false
- author: Dario Arcos-Díaz
- categories: [healthcare, eda, visualization]
- image: images/medicare-drug-cost_thumb.png

Health care systems world-wide are under pressure due to the high costs associated with disease. Now more than ever, particularly in developed countries, we have access to the latest advancements in medicine. This contrasts with the challenge of making those treatments available to as many patients as possible. It is imperative to find ways maximize the positive impact on the quality of life of patients, while maintaining a sustainable health care system. For this purpose I performed an analysis of Medicare data in the USA. Furthermore I used a drug-disease open database to cluster the costs by disease. I identified the most expensive diseases (mostly chronic diseases such as Diabetes) and the most expensive medicines. A drug for the treatment of HCV infections (Harvoni) stands out with the highest total costs in 2015. After this first exploration, I propose the in-depth analysis of further data to enable more targeted conclusions and recommendations to improve health care, such as linking of price databases to compare drug costs for the similar indications or the analysis of population data registers that document life style characteristics of healthy and sick individuals to identify those at risk of developing high-cost diseases.

## Relevance

Health care costs amount to a considerable part of the national budgets all over the world. In 2015, $3.2 trillion were spent for health care in the USA ([17.8% of its GDP](https://www.cms.gov/research-statistics-data-and-systems/statistics-trends-and-reports/nationalhealthexpenddata/nationalhealthaccountshistorical.html)).  In Germany, the health care spending reached [11.3% of GDP in 2014](http://data.worldbank.org/indicator/SH.XPD.TOTL.ZS?locations=DE). On the one hand, this high health care costs can be explained by the population growth, particularly the elderly proportion, requiring higher investments to secure quality of life. On the other hand, new medicines are continously being discovered enabling the treatment of diseases that were once a sentence of death. This has as a consequence that many once fatal diseases have now become chronic with a high burden on the health care costs.

But how can governments and insurers make sure that patients receive the care they need, including latest technology advances, without bankrupting the system? One first step is the identification of high-cost diseases and drugs. This insights can then be used to identify population segments at high-risk of developing a disease, who can then be the focus of prevention measures.

Governments, insurers, patient organizations, pharmaceutical and biotech companies need all to leverage their available data, if we are to improve the health of patients now and in future generations.

## Methods

### Data sources

- [Medicare Drug Spending Data 2011-2015](https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Information-on-Prescription-Drugs/2015MedicareData.html): drug spending and utilization data. _In this analysis only Medicare Part D drugs were considered (drugs patients generally administer themselves)_
- [Therapeutic Targets Database](http://bidd.nus.edu.sg/BIDD-Databases/TTD/TTD_Download.asp): Drug-to-disease mapping with ICD identifiers.

### Tools

- pandas for data crunching
- fuzzywuzzy for fuzzy logic matching
- git for version control

### Data preprocessing

First, I cleaned up and processed the drug spending data available from Medicare for the years 2011-2015. This data includes the total spending, claim number, and beneficiary number --among others-- for each drug identified by its brand and generic names.



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('Paired')
sns.set_style('whitegrid')
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
```


```python
data = pd.read_csv('data/medicare_data_disease.csv')
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Brand Name</th>
      <th>Generic Name</th>
      <th>Claim Count</th>
      <th>Total Spending</th>
      <th>Beneficiary Count</th>
      <th>Total Annual Spending Per User</th>
      <th>Unit Count</th>
      <th>Average Cost Per Unit (Weighted)</th>
      <th>Beneficiary Count No LIS</th>
      <th>Average Beneficiary Cost Share No LIS</th>
      <th>Beneficiary Count LIS</th>
      <th>Average Beneficiary Cost Share LIS</th>
      <th>Year</th>
      <th>Matched Drug Name</th>
      <th>Indication</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>10 wash</td>
      <td>sulfacetamide sodium</td>
      <td>24.0</td>
      <td>1569.19</td>
      <td>16.0</td>
      <td>98.074375</td>
      <td>5170.0</td>
      <td>0.303518</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2011</td>
      <td>sulfacetamide</td>
      <td>Acne vulgaris</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1st tier unifine pentips</td>
      <td>pen needle, diabetic</td>
      <td>2472.0</td>
      <td>57666.73</td>
      <td>893.0</td>
      <td>64.576405</td>
      <td>293160.0</td>
      <td>0.196766</td>
      <td>422.0</td>
      <td>42.347204</td>
      <td>471.0</td>
      <td>7.54586</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1st tier unifine pentips plus</td>
      <td>pen needle, diabetic</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>60pse-400gfn-20dm</td>
      <td>guaifenesin/dm/pseudoephedrine</td>
      <td>12.0</td>
      <td>350.10</td>
      <td>11.0</td>
      <td>31.827273</td>
      <td>497.0</td>
      <td>0.704427</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2011</td>
      <td>pseudoephedrine</td>
      <td>Nasal congestion</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>8-mop</td>
      <td>methoxsalen</td>
      <td>11.0</td>
      <td>9003.26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>298.0</td>
      <td>30.212282</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2011</td>
      <td>methoxsalen</td>
      <td>Cutaneous T-cell lymphoma</td>
    </tr>
  </tbody>
</table>
</div>



I also processed the data from the Therapeutic Targets Database, which presents the indications (diseases) associated with a drug generic name. 


```python
diseases = pd.read_csv('data/drug-disease_keys.csv')
diseases.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>TTDDRUGID</th>
      <th>LNM</th>
      <th>Indication</th>
      <th>ICD9</th>
      <th>ICD10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>DAP000001</td>
      <td>quetiapine</td>
      <td>Schizophrenia</td>
      <td>295, 710.0</td>
      <td>F20, M32</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>DAP000002</td>
      <td>theophylline</td>
      <td>Chronic obstructive pulmonary disease</td>
      <td>490-492, 494-496</td>
      <td>J40-J44, J47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>DAP000003</td>
      <td>risperidone</td>
      <td>Schizophrenia</td>
      <td>295, 710.0</td>
      <td>F20, M32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>DAP000004</td>
      <td>dasatinib</td>
      <td>Chronic myelogenous leukemia</td>
      <td>205.1, 208.9</td>
      <td>C91-C95, C92.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>DAP000004</td>
      <td>dasatinib</td>
      <td>Solid tumours; Multiple myeloma</td>
      <td>140-199, 203.0, 210-229</td>
      <td>C00-C75, C7A, C7B, C90.0, D10-D36, D3A</td>
    </tr>
  </tbody>
</table>
</div>



Then, I used a fuzzy logic algorithm to match each drug generic name of the Medicare data with the closest element from the Therapeutic Targets Database. After having a list of exact matches, I assigned the first associated indication to each Medicare drug. For details on how I did this, please check [my github repository](https://github.com/dariodata/medicare-drug-cost/blob/master/data_preparation.ipynb).

## Results

### Figure 1: Most expensive drugs and indications by total spending in a 5-year interval



```python
spending = data.groupby('Indication').sum().sort_values(by='Total Spending', ascending=False)
spending.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Claim Count</th>
      <th>Total Spending</th>
      <th>Beneficiary Count</th>
      <th>Total Annual Spending Per User</th>
      <th>Unit Count</th>
      <th>Average Cost Per Unit (Weighted)</th>
      <th>Beneficiary Count No LIS</th>
      <th>Average Beneficiary Cost Share No LIS</th>
      <th>Beneficiary Count LIS</th>
      <th>Average Beneficiary Cost Share LIS</th>
      <th>Year</th>
    </tr>
    <tr>
      <th>Indication</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Diabetes mellitus</th>
      <td>1619475</td>
      <td>367700954.0</td>
      <td>5.360758e+10</td>
      <td>73949096.0</td>
      <td>562815.013217</td>
      <td>2.494315e+10</td>
      <td>7277.339052</td>
      <td>40704222.0</td>
      <td>96730.900344</td>
      <td>33243845.0</td>
      <td>6840.136642</td>
      <td>1449360</td>
    </tr>
    <tr>
      <th>Schizophrenia</th>
      <td>589890</td>
      <td>120108011.0</td>
      <td>3.029475e+10</td>
      <td>16787537.0</td>
      <td>665302.614794</td>
      <td>5.232045e+09</td>
      <td>23737.078668</td>
      <td>4132330.0</td>
      <td>75431.535074</td>
      <td>12655011.0</td>
      <td>2733.240495</td>
      <td>503250</td>
    </tr>
    <tr>
      <th>Chronic obstructive pulmonary disease</th>
      <td>260320</td>
      <td>85571788.0</td>
      <td>2.668149e+10</td>
      <td>18010399.0</td>
      <td>117832.051227</td>
      <td>5.169355e+09</td>
      <td>1525.772351</td>
      <td>8891428.0</td>
      <td>16115.553801</td>
      <td>9118941.0</td>
      <td>953.965398</td>
      <td>201300</td>
    </tr>
    <tr>
      <th>Pain</th>
      <td>2076820</td>
      <td>449297282.0</td>
      <td>2.237135e+10</td>
      <td>125509481.0</td>
      <td>635933.153421</td>
      <td>3.582438e+10</td>
      <td>8484.565888</td>
      <td>68195577.0</td>
      <td>86753.860239</td>
      <td>57310277.0</td>
      <td>5810.508368</td>
      <td>1660725</td>
    </tr>
    <tr>
      <th>Hypertension</th>
      <td>1241330</td>
      <td>659834372.0</td>
      <td>2.140793e+10</td>
      <td>127524840.0</td>
      <td>453862.885999</td>
      <td>3.924869e+10</td>
      <td>2875.275230</td>
      <td>84758076.0</td>
      <td>99376.738594</td>
      <td>42766156.0</td>
      <td>8392.438238</td>
      <td>1338645</td>
    </tr>
  </tbody>
</table>
</div>




```python
spending_drug = data.groupby('Brand Name').sum().sort_values(by='Total Spending', ascending=False)
spending_drug.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Claim Count</th>
      <th>Total Spending</th>
      <th>Beneficiary Count</th>
      <th>Total Annual Spending Per User</th>
      <th>Unit Count</th>
      <th>Average Cost Per Unit (Weighted)</th>
      <th>Beneficiary Count No LIS</th>
      <th>Average Beneficiary Cost Share No LIS</th>
      <th>Beneficiary Count LIS</th>
      <th>Average Beneficiary Cost Share LIS</th>
      <th>Year</th>
    </tr>
    <tr>
      <th>Brand Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lantus/lantus solostar</th>
      <td>10560</td>
      <td>40959410.0</td>
      <td>1.419734e+10</td>
      <td>7627126.0</td>
      <td>9059.358978</td>
      <td>7.905816e+08</td>
      <td>87.058010</td>
      <td>3685935.0</td>
      <td>1757.877943</td>
      <td>3941191.0</td>
      <td>119.054297</td>
      <td>10065</td>
    </tr>
    <tr>
      <th>nexium</th>
      <td>13330</td>
      <td>37338541.0</td>
      <td>1.129409e+10</td>
      <td>6968266.0</td>
      <td>8159.874433</td>
      <td>1.624007e+09</td>
      <td>35.029401</td>
      <td>2808631.0</td>
      <td>1282.120804</td>
      <td>4159635.0</td>
      <td>108.193170</td>
      <td>10065</td>
    </tr>
    <tr>
      <th>crestor</th>
      <td>4670</td>
      <td>43304032.0</td>
      <td>1.084924e+10</td>
      <td>8312848.0</td>
      <td>6460.499385</td>
      <td>1.917524e+09</td>
      <td>27.902605</td>
      <td>5275631.0</td>
      <td>1477.330224</td>
      <td>3037217.0</td>
      <td>126.576936</td>
      <td>10065</td>
    </tr>
    <tr>
      <th>advair diskus</th>
      <td>385</td>
      <td>30806126.0</td>
      <td>1.036056e+10</td>
      <td>7096159.0</td>
      <td>7313.316443</td>
      <td>2.273054e+09</td>
      <td>22.805665</td>
      <td>3613170.0</td>
      <td>1284.084879</td>
      <td>3482989.0</td>
      <td>98.586905</td>
      <td>10065</td>
    </tr>
    <tr>
      <th>abilify</th>
      <td>45</td>
      <td>12506518.0</td>
      <td>9.434570e+09</td>
      <td>1861785.0</td>
      <td>25165.999223</td>
      <td>3.818410e+08</td>
      <td>127.685877</td>
      <td>333884.0</td>
      <td>2541.346071</td>
      <td>1527901.0</td>
      <td>101.829169</td>
      <td>10065</td>
    </tr>
  </tbody>
</table>
</div>




```python
n_top = 40
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(8,8))
g = sns.barplot(x='Total Spending', y='Indication', data=spending.reset_index()[:n_top], estimator=np.sum, ax=ax1, 
                color=sns.xkcd_rgb['dodger blue'])
g.set(yticklabels=[i[:27] for i in spending[:n_top].index])
g.set_xlabel('Total Spending $')
g2 = sns.barplot(x='Total Spending', y='Brand Name', data=spending_drug.reset_index()[:n_top], estimator=np.sum, ax=ax2,
                 color='lightblue')
g2.set(yticklabels=[i[:20] for i in spending_drug[:n_top].index])
g2.set_xlabel('Total Spending $')
#plt.title('Top 50 indications by Beneficiary Count Sum from 2011 to 2015')
fig.suptitle('Top %s indications and drugs for 5-year total spending 2011-2015' %n_top, size=16)
plt.tight_layout()
fig.subplots_adjust(top=0.94)
plt.savefig('Top_%s_disease_drug.png' %n_top, dpi=300, bbox_inches='tight')
```


    
![png](/images/2017-02-06-medicare-drug-cost_files/2017-02-06-medicare-drug-cost_10_0.png)
    


#### Indications (left part)

A look at the total spending for the 5-year period 2011-2015 reveals that the bulk of drug spending is covered by a 
small set of diseases/indications (left graph). The total spending per indication decreases rapidly by going down the 
list of drugs.

Diabetes occupies the first place in this list with a total 5-year spending exceding $50 billion. Following in the 
list, we find other chronic diseases such as schizophrenia, chronic obstructive pulmonary disease, hypertension (high
 blood pressure), hypercholesterolemia (high cholesterol), depression, hiv infections, multiple sclerosis, peptic 
 ulcer disease, and chronic HCV infection (hepatitis C). Interestingly, pain medications are also in the top 4 
 indications by total spending.
 
It makes sense that treatment of chronic diseases receives the highest investment in drug spending, as patients with 
these diseases can live long lives when medicated.

Interestingly, the first cancer reaches only the 19th place of this list (chronic myelogenous leukemia). However, it 
must be noted that _cancer is actually a collection of different diseases_ with different genetics, origin, and 
treatment options. These different cancers were not clustered in this analysis.

#### Drugs (right part)

When we look at the most expensive drugs for the total 5-year spending, we find on the top of the list: Lantus 
(insulin), nexium (peptic ulcer), and crestor(anti cholesterol). It makes sense as these are medicines to treat chronic 
diseases.

However, we cannot learn much on a high level from looking at the total spending only. Therefore, a closer look is 
needed.


### Figure 2: Drug spending is growing but at very heterogeneous rates


```python
spend_2015_ind = data[data['Year']==2015].groupby('Indication').sum().sort_values(by='Total Spending', ascending=False)
#spend_2015_drug = data[data['Year']==2015].groupby('Brand Name').sum().sort_values(by='Total Spending', 
# ascending=False)
spend_2015_ind.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Claim Count</th>
      <th>Total Spending</th>
      <th>Beneficiary Count</th>
      <th>Total Annual Spending Per User</th>
      <th>Unit Count</th>
      <th>Average Cost Per Unit (Weighted)</th>
      <th>Beneficiary Count No LIS</th>
      <th>Average Beneficiary Cost Share No LIS</th>
      <th>Beneficiary Count LIS</th>
      <th>Average Beneficiary Cost Share LIS</th>
      <th>Year</th>
    </tr>
    <tr>
      <th>Indication</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Diabetes mellitus</th>
      <td>323895</td>
      <td>80808515.0</td>
      <td>1.538882e+10</td>
      <td>16756712.0</td>
      <td>179199.546237</td>
      <td>5.861327e+09</td>
      <td>2112.287818</td>
      <td>9688833.0</td>
      <td>24716.663299</td>
      <td>7067683.0</td>
      <td>1459.860044</td>
      <td>290160</td>
    </tr>
    <tr>
      <th>Chronic HCV infection</th>
      <td>5421</td>
      <td>272915.0</td>
      <td>8.349020e+09</td>
      <td>90487.0</td>
      <td>182144.098903</td>
      <td>7.546096e+06</td>
      <td>2134.680173</td>
      <td>30454.0</td>
      <td>10701.886971</td>
      <td>60033.0</td>
      <td>156.351475</td>
      <td>4030</td>
    </tr>
    <tr>
      <th>Chronic obstructive pulmonary disease</th>
      <td>52064</td>
      <td>17764181.0</td>
      <td>6.756824e+09</td>
      <td>3803356.0</td>
      <td>30714.837787</td>
      <td>1.067257e+09</td>
      <td>561.945076</td>
      <td>1931374.0</td>
      <td>4177.371552</td>
      <td>1871982.0</td>
      <td>232.217985</td>
      <td>40300</td>
    </tr>
    <tr>
      <th>Schizophrenia</th>
      <td>117978</td>
      <td>25030047.0</td>
      <td>5.468897e+09</td>
      <td>3493417.0</td>
      <td>192134.968549</td>
      <td>1.084651e+09</td>
      <td>8292.122415</td>
      <td>938911.0</td>
      <td>19320.941152</td>
      <td>2554333.0</td>
      <td>573.729378</td>
      <td>100750</td>
    </tr>
    <tr>
      <th>Pain</th>
      <td>415364</td>
      <td>94109025.0</td>
      <td>4.956161e+09</td>
      <td>27047833.0</td>
      <td>164077.757502</td>
      <td>7.597111e+09</td>
      <td>4357.883695</td>
      <td>15366894.0</td>
      <td>19810.346154</td>
      <td>11680708.0</td>
      <td>1105.913592</td>
      <td>332475</td>
    </tr>
  </tbody>
</table>
</div>




```python
top_10_spend = data[data['Year']==2015].sort_values(by='Total Spending', ascending=False)[['Brand Name', 
                                                                                           'Total Spending', 
                                                                                           'Year']][:10]
top_10_spend
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Brand Name</th>
      <th>Total Spending</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19770</th>
      <td>harvoni</td>
      <td>7.030633e+09</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>20104</th>
      <td>lantus/lantus solostar</td>
      <td>4.359504e+09</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>18926</th>
      <td>crestor</td>
      <td>2.883122e+09</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>18069</th>
      <td>advair diskus</td>
      <td>2.270016e+09</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>21640</th>
      <td>spiriva</td>
      <td>2.191466e+09</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>19988</th>
      <td>januvia</td>
      <td>2.131952e+09</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>21404</th>
      <td>revlimid</td>
      <td>2.077425e+09</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>20658</th>
      <td>nexium</td>
      <td>2.012921e+09</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>20291</th>
      <td>lyrica</td>
      <td>1.766474e+09</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>19818</th>
      <td>humira/humira pen</td>
      <td>1.662292e+09</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(8,5))

g=sns.factorplot(x='Year', y='Total Spending', hue='Brand Name', palette='coolwarm', 
                 hue_order=top_10_spend['Brand Name'],
                 data=data[data['Brand Name'].isin(top_10_spend['Brand Name'])], ax=ax1)
ax1.set_title('Annual spending for top 10 drugs')
ax1.set_ylabel('Total Spending $')
plt.close(g.fig)

ax2.scatter(x=spend_2015_ind['Beneficiary Count'][:100], 
            y=spend_2015_ind['Total Spending'][:100],
            s=spend_2015_ind['Claim Count'][:100]/100000,
            #c=spend_2015_ind.reset_index()['Indication'][:100])
            color=sns.xkcd_rgb['dodger blue'], alpha=0.75)
ax2.set_title('Top 100 indications in 2015')
plt.xlabel('Beneficiary Count')
plt.ylabel('Total Spending $')
plt.axis([0, None, 0, None])
for label, x, y in zip(spend_2015_ind.index, 
                       spend_2015_ind['Beneficiary Count'][:10], 
                       spend_2015_ind['Total Spending'][:10]):
    plt.annotate(label, xy=(x, y), color='red', alpha=0.75)
fig.suptitle('Annual drug spending development and overview of highest-cost indications', size=16)
plt.tight_layout()
fig.subplots_adjust(top=0.85)
plt.savefig('Top_bubble_disease_drug.png', dpi=300, bbox_inches='tight')
```


    
![png](/images/2017-02-06-medicare-drug-cost_files/2017-02-06-medicare-drug-cost_15_0.png)
    


#### Annual spending development for top 10 drugs (left)

The drug landscape is not temporally static. Therefore, I analyzed the annual spending since 2011 for the 10 top drugs
 in 2015. Eight out of these ten drugs consistently received higher spending every year, a reflection of the general 
 health care spending panorama. However, the rate of growth for each drug is dramatically different. Particularly 
 striking is the case of the drug Harvoni, which exhibited a >7-fold growth in total spending between 2014 and 2015.
 
Harvoni is a medicine for the treatment of hepatitis C (HCV infection) that was launched in 2014. It is the first 
drug with _cure_ rates close to 100%. Harvoni practically cures a chronic disease and this is reflected in its 
pricing at over $90k for a 12 week treatment.

The remaining drugs in the figure are mostly used for the _treatment_ of chronic diseases.

But how can we more extensively evaluate the burden posed by the different diseases/indications?

#### Top 100 indications in 2015 (right)

In order to find out more about the distribution of the most expensive indications, I plotted the drug spendings 
grouped by indication for the year 2015 in a scatter plot. This way, we can not only look at the total spending but 
also at the number of beneficiaries for a particular indication. The size of the bubbles represents the relative number
 of claims. 
 
 From this graph we can assess the magnitude of how the most expensive diseases affect society. Diabetes is not only 
 the most expensive single indication by total spending but also affects a very large number of people.
 
 The indications with the most beneficiaries are hypertension, pain and high cholesterol. They also represent some of 
 the highest number of claims (bubble size). This indicates that the average cost associated with each claim is low, 
 as these are generally medications with expired patents that are priced very low.
 
Again it is interesting to take a look at chronic HCV infection. This is the indication for the drug Harvoni. Both 
the number of beneficiaries and claims are extremely low compared with other diseases. However, chronic HCV infection
 reached the second place in the highes total drug spending in 2015.

## Next steps

I have shown in this analysis that very interesting insights can be gained from analyzing a smaller set of publicly 
available data. It follows that a more detailed and deeper analysis could enable more targeted conclusions and 
recommendations for improving the health care system and the quality of life of patients suffering from disease.  
Access to non-public owned data would make even deeper analysis possible.

Additional analysis could include:

- Clustering of diseases/indications to higher-level categories (cancer, metabolic disease, circulatory disease, 
nervous system disease, etc.)
- Linking of price databases to compare drug costs for the same indication on a population level
- Analysis of population data registers that document life style characteristics of healthy and ill individuals to 
identify those at risk of developing high-cost diseases (e.g.
 [Medical Expenditure Panel Survey](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-170), 
 [Behavioral Risk Factor Surveillance System data](https://www.cdc.gov/brfss/annual_data/annual_2015.html)) 


## Limitations

One limitation from this analysis is that only Part D drugs were considered. A further analysis could include Part B drugs too.

Moreover it was assumed that the fuzzy logic matching was successful in most cases. A more detailed test is required 
to assess match success more stringently.

All conclusions are only valid for the 2011-2015 interval. No data for 2016 was analyzed.

