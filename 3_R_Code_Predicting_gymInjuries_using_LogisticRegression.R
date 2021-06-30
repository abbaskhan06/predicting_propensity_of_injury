# 
# A Binary Classification Model on Predicting the probability of an Injury occurring in the Gym (Logistic Regression)
# 

#--------------------------------------------
# Step 01 - Installing Required Packages & Loading Libraries
#--------------------------------------------

#install.packages('caTools')
#install.packages('dummies')
# install.packages('dplyr')
# install.packages('ROCR')

library(caTools)
library(dummies)
library(dplyr)
library(ROCR)



#--------------------------------------------
# Step 02 - Importing the Dataset and Processing for Analysis
#--------------------------------------------


#Setting the Work Directory
setwd("G:/Customer_Analysis_ACC")


# Importing the Dataset from CSV
gymInjury = read.csv(file='DataForExercise.csv', header = TRUE, sep = ',')


# Data Audit and Sanity Checks


# Data Sanity check 01
# View the data set (Top 10 Rows of Each Variable)
head(gymInjury,10)


# Data Sanity check 02
# Look at Contents of the data set
str(gymInjury)


# Here we can see that the y variable is whether the person experience a gym injury in the year following the extraction date?
# This variable would indicate the probability of an injury happening at the gym given the values of other variables.
# Hence this would be our binary dependent variable
# The y variable has both yes / no. But are they sizable?


# Data Sanity check 03
# Checking the frequency of Y and N values within the "y" variable 
table(gymInjury$y)


# We can see that the Y value has a penetration of 5% (4000/80000)


# Data Sanity check 04
# Checking the presence of missing values within the Dataset 
# This is done by creating a new table devoid of Missing Values  
gymInjury_no_missing = na.omit(gymInjury) 

# The new Dataset now has 46880 records (58.6%) of the original Dataset



# Data Pre-Processing 

# Variable Creation

# Creating dependent variable 'Y' from the values of yes/no of the 'y' variable in the dataset as the dependent variable needs a binomial value

gymInjury$Y = ifelse(gymInjury$y =='Y',1,0)

# Dropping Unwanted Variables acci_year and the original y variable
gymInjury = subset(gymInjury, select = -c(acci_year,y))


# Creating Dummy Variables for categorical Variables as Regression Model takes ONLY numeric values for independent variables 
gymInjury_d = dummy.data.frame(gymInjury, sep = '_')

#Checking Variable Names
names(gymInjury_d)

# Replacing space in the column name with _ (For Example in 'Location_tla_last_claim_Auckland City')
colnames(gymInjury_d) = gsub(" ", "_",colnames(gymInjury_d))

#Checking Variable Names
names(gymInjury_d)

# We now have variable 'location_tla_last_claim_Auckland_City' instead of 'Location_tla_last_claim_Auckland City' 


# Dropping Unwanted variable 'work_last_claim_NA' which got created while creating the Dummy Variable
gymInjury_d = subset(gymInjury_d, select = -c(work_last_claim_NA))



# Missing Value Treatment 


# Replacing Missing Values in each variable within the Dataset using the Median Values of the corresponding variable
gymInjury_imp = data.frame(
        sapply(
                gymInjury_d,
                function(x) ifelse(is.na(x),
                                   median(x, na.rm = TRUE),
                                   x)))

#Since age_at_extraction_date variable has negative values as well as 0 values, we need to run another function to replace them with median values
gymInjury_imp$age_at_extraction_date = ifelse(gymInjury_imp$age_at_extraction_date< 0 & 
                                                gymInjury_imp$age_at_extraction_date ==0,
                                            median(gymInjury_imp$age_at_extraction_date),
                                            gymInjury_imp$age_at_extraction_date)




#--------------------------------------------
# Step 03 - Splitting Dataset into Train and Test
#--------------------------------------------

#Setting Seed Number so that we have the same records for Test and Train data during each subsequent run 
set.seed(123)

#Splitting the Train:Test by 75:25
split = sample.split(gymInjury_imp$Y, SplitRatio = 0.75)

train_set = subset(gymInjury_imp, split == TRUE)

test_set = subset(gymInjury_imp, split == FALSE)



#--------------------------------------------
# Step 04 - Running Model on Train Dataset and Variable Reduction
#--------------------------------------------


# Running the first instance of Logistic Regression Model  

myLogit = glm(Y~.,data = train_set, family=binomial)
summary(myLogit)

# The Model Results are not good as we are getting values with NA and all other variables are getting significant p values.

# The Training Data needs to be optimised through the "StepWise" variable reduction process and then rerun for Logistic Output

# We can also check for Multi Collinearity between the dependent variables, but these would partially be taken care by the "StepWise" function 

# Also certain variables need to be removed before the "StepWise" procedure such as 
  # Variables with Estimates as NA ('location_tla_last_claim_Whangarei_District' & 'ethnicity_last_claim_Residual_Categories')
  # Variables representing Member ID ('PersonACCId')
  # Variables from which no insights can be derived ('location_tla_last_claim_Unknown')



# Re Running the Logistic Model on the Training Dataset with the above 4 variables removed   
myLogit = glm(formula = Y ~ . 
              -PersonACCId 
              -location_tla_last_claim_Whangarei_District 
              -ethnicity_last_claim_Residual_Categories
              -location_tla_last_claim_Unknown
              , 
              data = train_set, family=binomial)
summary(myLogit)


# Running "StepWise Regression" on the above Logistic Model for Variable Reduction            
myLogit_step = step(myLogit)

summary(myLogit_step)


# Improving the model results by manually removing insignificant variables from the output of StepWise 
myLogit_step = glm(formula = Y ~  
                      age_at_extraction_date
                    + ethnicity_last_claim_European
                    + location_tla_last_claim_Auckland_City
                    + location_tla_last_claim_North_Shore_City
                    + location_tla_last_claim_Wellington_City
                    + work_last_claim_Heavy_Work
                    + work_last_claim_Light_Work
                    + work_last_claim_Medium_Work
                    + work_last_claim_Sedentary_Work
                    + Areaunit_score
                    + num_gym_all
                    + num_wgt_all
                    + back_sprain_all
                    + kneeleg_sprain_all
                    + soft_tissue_all
                    + lower_back_all
                    + upper_back_spine_all
                    
                    , data = train_set, family = binomial)

summary(myLogit_step)




#--------------------------------------------
# Step 05 - Creating Final Model Summary
#--------------------------------------------


# Taking the Summary Coefficients of the Model Results into a variable (summary.coeff0) 
summary.coeff0 = summary(myLogit_step)$coefficient


#Calculating Odd Ratios for the selected variables and merging them with the summary coefficients

OddRatio = exp(coef(myLogit_step))
summary.coeff = cbind(Variable = row.names(summary.coeff0), OddRatio, summary.coeff0)
row.names(summary.coeff) = NULL

#R Function : Standardised Coefficients
# Standardised Coefficients are transformed values of Estimates of the corresponding variable within the model
# This is done so as to bring all variables on a common scale and compare their contribution to the overall model 

stdz.coff = function (regmodel) 
  { b = summary(regmodel)$coef[-1,1]

  sx = sapply(regmodel$model[-1], sd)
beta = (3^(1/2))/pi * sx * b
return(beta)
}

std.Coeff = data.frame(Standardized.Coeff = stdz.coff(myLogit_step))
std.Coeff = cbind(Variable = row.names(std.Coeff), std.Coeff)
row.names(std.Coeff) = NULL

#Final Summary Report
final = merge(summary.coeff, std.Coeff, by = "Variable", all.x = TRUE)

# Exporting the Summary Table
write.csv(final,"final_summary.csv")




#--------------------------------------------
# Step 06 - Using Model Results from Train Data to Predict Values on Test Data 
#--------------------------------------------


# Predicting the Probability of Y variable on Test_Set
prob_pred = predict(myLogit_step,test_set, type = 'response')

# Merging Probability Values with Test_Set 
final_data = cbind(test_set, prob_pred) 




#--------------------------------------------
# Step 07 - Calculating Model Diagnostics on the Test_Set using ROCR package
#--------------------------------------------

library(ROCR)
# https://www.r-bloggers.com/2014/12/a-small-introduction-to-the-rocr-package/

# Creating Prediction Instance to Calculate Model Diagnostics 
pred_val = prediction(prob_pred ,final_data$Y)


# Calculating Maximum Accuracy and Prob cutoff against it 

acc.perf = performance(pred_val, "acc") 
ind = which.max( slot(acc.perf, "y.values")[[1]]) 
acc = slot(acc.perf, "y.values")[[1]][ind]
cutoff = slot(acc.perf, "x.values")[[1]][ind]

# Print Results
print(c(accuracy= acc, cutoff = cutoff))


# Calculating Area under Curve
perf_val = performance(pred_val,"auc")
# Print Results
print(c("AreaUnderROCcurVe(AUC)"= perf_val@y.values) )


# Plotting Lift curve
plot(performance(pred_val, measure="lift", x.measure="rpp"), colorize=TRUE)


# Plot the ROC curve
perf_val2 <- performance(pred_val, "tpr", "fpr")
plot(perf_val2, col = "green", lwd = 1.5)


#Calculating KS statistics
ks1.tree <- max(attr(perf_val2, "y.values")[[1]] - (attr(perf_val2, "x.values")[[1]]))
ks1.tree

# Making the confusion matrix
y_pred = ifelse(prob_pred > 0.5, 1, 0)
cm = table(test_set[, 45], y_pred)
cm



#--------------------------------------------
# Step 08 - Assigning Probability of Occurrence of an Injury in Gym to the Entire Dataset
#--------------------------------------------

# Predicting the Probability of Y variable on Entire Dataset (All Members)
prob_pred2 = predict(myLogit_step,gymInjury_imp, type = 'response')

# Assigning Predicted Probability and Deciles to all members
final_data_temp = cbind(gymInjury_imp, prob_pred) 
final_data2 = mutate(final_data_temp, decile = ntile(desc(prob_pred),10))
final_data3 = subset(final_data2, select = c(PersonACCId,prob_pred,decile))
final_data4 = filter(final_data3, Y==1)

# Exporting the data of ACC members  
write.csv(final_data4,"ACCId_Probability.csv")
write.csv(final_data3,"Injury.csv")


#
#
#

#--------------------------------------------
# END OF CODE
#--------------------------------------------


