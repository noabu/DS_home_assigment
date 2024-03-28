# DS_home_assigment
## task's goal:
conducting an Exploratory Data Analysis (EDA) on the provided dataset, CICIDS2017. The objective is to extract three notable insights from the data, utilizing both Machine Learning (ML) or Deep Learning (DL) techniques and a recommendation system, tailored to my preferences.

## work process:
1. Import the dataset and understand what it represents and what the features represent:
The dataset includes network traffic, encompassing both malicious attacks and normal behavior. It comprises eight distinct CSV files, each focusing on a unique attack type: Denial of Service (DoS), Distributed Denial of Service (DDoS), Web Attack, PortScan, Botnet activity, Infiltration and Brute Force attacks. The dataset was collected over a period of 5 days.
Each row in the dataset represents a network flow (which means a sequence of packets that share common characteristics such as source and destination IP address, source and destination PORT, protocol type, etc.). total rows:  2,830,743.
each column is a feature that presents various aspects of network traffic and attacks. total features: 77.
some of the features represent packets length, packets amount, flow duration, interval arrival time of packets, Flags of TCP etc.
Additionally, I attempted to discern which features within the dataset could characterize different types of attacks. For instance, in the case of Denial of Service (DoS) attacks, the volume of transmitted packets and the duration of the flow may signify suspicious traffic. Similarly, for Brute Force attacks, characteristics such as the quantity of packets sent in both directions, packet lengths, and flow duration can reveal suspicious patterns. Notably, some of these insights were later corroborated by the classifier I employed.
2.Preprocess the data:
   * Split the data into features an target. (in function: split_data_to_features_and_target)
   * Preprocess the labels: replace the labels to numerical (in function: target_to_numeric)
   * Preprocess the features data:
   * I opted to merge the CSV files into a single dataset to simplify the process and train a single classifier. However, I acknowledged that this approach might lead to reduced accuracy and overlook important features relevant to certain attacks. before I concat them I checked that the includes the same features and the same datatypes (in function: check_if_can_concat).
   * Checked if there are non numerical features (there wasn't).
   * Checked for NaN values - the Flow Bytes feature contained missing values (NaN), prompting me to determine suitable replacements. Given that the traffic data spans five consecutive days, I opted for linear interpolation to minimize information loss. (in function: features_preprocessing)
   * Replace Infinite values in the Max value of the column . (in function: features_preprocessing).
3. Visualizations & Features reduce: (in functions: data_visualization, remove_low_variance_features, reduce_features_with_high_correlation, normalize_features)
   To gain a deeper understanding of the data, I visualized it using Matplotlib by plotting histograms for each feature and generating a correlation matrix. Initially, I plotted histograms for all features and utilized the correlation matrix to identify candidates for reduction, considering the risk of overfitting due to the large number of features. I filtered out features that had a constant value of 0 across all rows and those with high correlation with other features. Total features after filtering: 17.
   features that I use: ['Destination Port', 'Fwd Packet Length Min', 'Bwd Packet Length Min', 'Flow Bytes/s', 'Flow IAT Min', 'Fwd Header Length', 'Bwd Header Length', 'Bwd Packets/s',
   'Min Packet Length', 'PSH Flag Count', 'ACK Flag Count', 'Down/Up Ratio', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'min_seg_size_forward', 'Active Std', 'Idle Std']
HeatMap of all the provided features :
![image](https://github.com/noabu/DS_home_assigment/assets/37350541/288de439-9493-4775-abba-4fa67e6155bb)
   
The subsequent images are histograms of the filtered features, providing a more manageable subset for analysis:
![image](https://github.com/noabu/DS_home_assigment/assets/37350541/4df6565c-2bf1-435f-8eb0-46a952dc3177)
![image](https://github.com/noabu/DS_home_assigment/assets/37350541/cb5b57e5-ac19-4d1e-96a8-f0d22a35ece9)
![image](https://github.com/noabu/DS_home_assigment/assets/37350541/68df9239-0654-4fd3-b130-3cd593dcc099)
from this we can learn that the features are in different ranges (that why I chose to normlize the features) - some 0/1 (flags) some 0-500 some 0-120,000,000 etc. 
Also, we can learn that there are peaks in the data which can indicate the behavior of the flow.

In addition I visualized the distribution of the target classes:
![image](https://github.com/noabu/DS_home_assigment/assets/37350541/e31971d3-ff2b-41e4-8e94-5fa03909b0d3)
From this we can learn that the common attacks in this dataset are: DoS, DDoS and PortScan. in addition we learn the classes are not balance.










