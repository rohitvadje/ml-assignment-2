# ml-assignment-2
Machine Learning Assignment 2 – Classification Models

---------------------------------------------------------------------------------------------------------------------------------------------
Name: Rohit Vadje
Course: M.Tech AIML
Assignment: ML Assignment 2
---------------------------------------------------------------------------------------------------------------------------------------------

a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether a tumor is malignant or benign based on medical features. The project demonstrates the complete machine learning workflow including model training, evaluation, UI development using Streamlit, and cloud deployment.


b. Dataset Description

The dataset used is the Breast Cancer Wisconsin Dataset available from the Scikit-learn built-in repository.

Dataset Details:
	•	Number of Instances: 569
	•	Number of Features: 30 numerical features
	•	Target Classes: Binary Classification (Malignant / Benign)
	•	No missing values
	•	Suitable for classification tasks and performance comparison.


c. Models Used and Evaluation Metrics

The following six machine learning models were implemented and evaluated using:
	•	Accuracy
	•	AUC Score
	•	Precision
	•	Recall
	•	F1 Score
	•	Matthews Correlation Coefficient (MCC)



Comparison Table
---------------------------------------------------------------------------------
ML Model Name	                  Accuracy	AUC	    Precision	Recall	F1	    MCC
---------------------------------------------------------------------------------
1) Logistic Regression	        0.956	    0.997	  0.946	    0.986	  0.966	  0.907
2) Decision Tree	              0.947	    0.944	  0.958	    0.958	  0.958	  0.888
3) kNN	                        0.956	    0.996	  0.934	    1.000	  0.966	  0.909
4) Naive Bayes	                0.974     0.998	  0.959	    1.000	  0.979	  0.945
5) Random Forest (Ensemble)	    0.965	    0.995	  0.959	    0.986	  0.972	  0.925
6) XGBoost (Ensemble)	          0.956	    0.991	  0.958	    0.972	  0.965	  0.906
---------------------------------------------------------------------------------



Model Performance Observations
---------------------------------------------------------------------------------------------------------------------------------------------
ML Model Name	              Observation about Model Performance
---------------------------------------------------------------------------------------------------------------------------------------------
Logistic Regression	        Provided very high accuracy and stable performance with excellent AUC. Works    well for linear separable data.
Decision Tree	              Slightly lower accuracy; prone to overfitting but easy to interpret.
kNN	                        Achieved perfect recall but slightly lower precision; sensitive to distance metrics and dataset scaling.
Naive Bayes	                Best overall performance with highest accuracy and MCC. Assumes feature independence but worked very efficiently.
Random Forest (Ensemble)    Strong performance due to ensemble learning; balanced precision and recall.
XGBoost (Ensemble)	        High performance with good generalization; slightly lower AUC than Naive Bayes but very robust.
---------------------------------------------------------------------------------------------------------------------------------------------



Conclusion:
Ensemble models and Naive Bayes performed best on this dataset. The project successfully demonstrates end-to-end ML workflow including model comparison, deployment, and UI interaction using Streamlit.
