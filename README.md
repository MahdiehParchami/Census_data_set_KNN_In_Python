# Census_data_set_KNN_In_Python
Implementing K Nearest Neighbors machine learning technique on Census data set

I used a given census data set, which has been derived from the US Census and 
determines whether a person's income is above or below 50K according to some features.
I want to perform the KNN machine learning model to predict income based 
on some other features like occupation, education, gender, and race and see how accurately we can 
classify low-income from high-income citizens.

In this study, I performed exploratory data analysis to get a better understanding of data, I discovered that people with Master, Doctorate, and prof-school degrees, self-employee work class, Exec-managerial, and prof-specialty occupations and married status earn more money compared to others. Next, I discovered there was not any noticeable correlation between the target variable income and the rest of the variables. Also, I ignored Capital gain and capital loss variables because they have more than 90% zero and I ignored the Fniwgt variable as well. Finally, I examined KNN machine learning models for both data set, with feature-selected data set and with keeping all variables, to predict income and compared the result. I found that the featured selected data set gave 
us better results. optimal K would be 10 with a 0.81 accuracy.
