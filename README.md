# DS_home_assigment
## task's goal:
conducting an Exploratory Data Analysis (EDA) on the provided dataset, CICIDS2017. The objective is to extract three notable insights from the data, utilizing both Machine Learning (ML) or Deep Learning (DL) techniques and a recommendation system, tailored to my preferences.

## work process:
1. Import the dataset and understand what it represents and what the features represent:<br>
The dataset encompasses network traffic containing both attack instances and benign behavior.<br> It comprises eight distinct CSV files, each focusing on a unique attack type: Denial of Service (DoS), Distributed Denial of Service (DDoS), Web Attack, PortScan, Botnet activity, Infiltration and Brute Force attacks. The dataset was collected over a period of 5 days.<br>
Each row in the dataset represents a network flow (which means a sequence of packets that share common characteristics such as source and destination IP address, source and destination PORT, protocol type, etc.). total rows:  2,830,743.<br>
each column is a feature that presents various aspects of network traffic and attacks. total features: 77.<br>
some of the features represent packets length, packets amount, flow duration, interval arrival time of packets, Flags of TCP etc.<br>
Additionally, I attempted to discern which features within the dataset could characterize different types of attacks. <br>For instance, in the case of Denial of Service (DoS) attacks, the volume of transmitted packets and the duration of the flow may signify suspicious traffic. <br>Similarly, for Brute Force attacks, characteristics such as the quantity of packets sent in both directions, packet lengths, and flow duration can reveal suspicious patterns. <br>Notably, some of these insights were later corroborated by the classifier I trained.<br>
2.Preprocess the data:<br>
   * Split the data into features an target. (related function: split_data_to_features_and_target)
   * Preprocess the labels: replace the labels to numerical (related function: target_to_numeric)
   * Preprocess the features data:
   <br> ** I chose to merge the CSV files into a single dataset to simplify the process and train a single classifier. However, I acknowledged that this approach might lead to reduced accuracy and overlook important features relevant to certain attacks.<br> Before I concat them I checked that they include the same features and the same datatypes (related function: check_if_can_concat).
   <br> ** Checked if there are non numerical features (there weren't).
   <br> ** Checked for NaN values - the Flow Bytes feature contained missing values (NaN), prompting me to determine suitable replacements. Given that the traffic data was collected during 5 days in a row, I chose linear interpolation to minimize information loss. (related function: features_preprocessing)
   <br> ** Replace Infinite values in the Max value of the column . (related function: features_preprocessing).
3. Visualizations & Features reduce: <br>(related functions: data_visualization, remove_low_variance_features, reduce_features_with_high_correlation, normalize_features)<br>
   To gain a deeper understanding of the data, I visualized it using Matplotlib by plotting histograms for each feature and generating a correlation matrix.<br>Initially, I plotted histograms for all features and utilized the correlation matrix to identify candidates for reduction, considering the risk of overfitting due to the large number of features.<br> I filtered out features that had a constant value of 0 across all rows and those with high correlation with other features. Total features after filtering: 17.
   features that I use:<br> [Destination Port, Fwd Packet Length Min, Bwd Packet Length Min, Flow Bytes/s, Flow IAT Min, Fwd Header Length, Bwd Header Length, Bwd Packets/s,
   Min Packet Length, PSH Flag Count, ACK Flag Count, Down/Up Ratio, Init_Win_bytes_forward, Init_Win_bytes_backward, min_seg_size_forward, Active Std, Idle Std]<br>
HeatMap of all the provided features :
![image](https://github.com/noabu/DS_home_assigment/assets/37350541/288de439-9493-4775-abba-4fa67e6155bb)
   
The subsequent images are histograms of the filtered features, providing a more manageable subset for analysis:
![image](https://github.com/noabu/DS_home_assigment/assets/37350541/4df6565c-2bf1-435f-8eb0-46a952dc3177)
![image](https://github.com/noabu/DS_home_assigment/assets/37350541/cb5b57e5-ac19-4d1e-96a8-f0d22a35ece9)
![image](https://github.com/noabu/DS_home_assigment/assets/37350541/68df9239-0654-4fd3-b130-3cd593dcc099)
from this we can conclude that the features are in different ranges (that's why I chose to normlize the features) - some 0/1 (flags) some 0-500 some 0-120,000,000 etc. 
Also, we can learn that there are peaks in the data which can indicate the behavior of the flow.

In addition I visualized the distribution of the target classes:
![image](https://github.com/noabu/DS_home_assigment/assets/37350541/e31971d3-ff2b-41e4-8e94-5fa03909b0d3)
From this we can conclude that the common attacks in this dataset are: DoS, DDoS and PortScan. in addition we learn the classes are not balanced.

## ML technique:
I chose the Random Forest classifier because of its adeptness in handling imbalanced classes. Its structure allows it to distribute focus across different classes, mitigating the risk of overfitting to the majority class. 
results:
accuracy: 0.988
precision: 0.82
recall: 0.84



## insights from Random Forest classifier:
1. The pivotal features within this dataset that may signal suspicious network flow include:
   <br>Destination Port, Init_Win_bytes_backward, Init_Win_bytes_forward, Bwd Packets/s, Bwd Packet Length Min, Idle Std, Flow IAT Min,
   <br>Bwd Header Length, Down/Up Ratio, Fwd Packet Length Min, Flow Bytes/s, PSH Flag Count, ACK Flag Count, Min Packet Length, Fwd Header Length, and Active Std.
   based on:
Classifier feature importances:
![image](https://github.com/noabu/DS_home_assigment/assets/37350541/dceba4ba-e209-4ba2-ad93-6a72cc40edff)
Detecting patterns in network traffic that include these features and exhibit abnormal behaviors can suggest a suspicious connection.
2. This insight focused on DDoS attack: <br>
   The smaller the port number, the greater the chance that it is a flow with suspicious behavior related yo DDoS attacks.
This can be seen directly from the Partial Dependence Plot (that created based on the classifier I trained):
![image](https://github.com/noabu/DS_home_assigment/assets/37350541/52dba98d-3b97-4874-984e-6b8804584183)

## insights using Linear Reggression:
I wanted to check my 2nd insight by perform linear regression.
because the CSV are splited by different type attack I chose to use only the dataframe that includes only DDoS attack. <br>
the linear regression results:<br>
Slope: -1.2786453377487048e-05 <br>
because that the slope is negative it point on a negative relationship (as X increases, Y decreases) means as the port number is bigger, the greater the chance that it is a flow with suspicious behavior related yo DDoS attack . <br>
This conclusion is the opposite from my 2nd insight. <br>
The reason I can think about its mabye cause the model trained over all the classes where in this I look the data only from the DDoS CSV.
![image](https://github.com/noabu/DS_home_assigment/assets/37350541/5aa3e1c5-a919-41f7-b0a6-7916585a7fa2)
So I perform linear regression over all CSV files:<br>
Slope: -1.1065431906928372e-05 <br>
Not as I expected here too the slope is negative.
![image](https://github.com/noabu/DS_home_assigment/assets/37350541/34be3ce6-340c-4f3f-a061-9e51df3f82ab)


## Future work:
1. Group the 14 types of attacks into 7 according to the division of the CSV files (or binary classifier- Attack/Benign) and see if there are other/more interesting insights.
2. To train a classifier for each type of attack separately to check if it is possible to learn from it about the type of each attack.
3. Continue to explore the contradiction of Insight 2 & Insight 3.











