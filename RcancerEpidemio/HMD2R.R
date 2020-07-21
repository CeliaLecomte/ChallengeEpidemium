#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Script File: HMD2R.R  
# Date of creation: 3 mars 2016
# Date of last modification: 3 mars 2016
# Author: Seraya Maouche <seraya.maouche@iscb.org>
# Project: Epidemium (http://www.epidemium.cc)
# Short Description: This script provides functionalities to import
#                     data from the Human Mortality Database (HMD).
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Install required package 
install.packages("demography")
install.packages("lifecontingencies")
install.packages("forecast")
install.packages("StMoMo")

# Load required packages
library(demography)

# Use the function hmd.mx() to connect to HMD (you need a username and password that you can obtain from HMD)
# HMD: http://www.mortality.org
# The parameter Country can be used to chose a dataset (38 datasets from different countries are available in HMD)
# Country code for France : France total population=FRATNP, France civilian population=FRACNP

FrenchDemogData <-hmd.mx(country="FRACNP", username="XXXXXXXXXXXXX", password="YHYUNH", label="French")
summary(FrenchDemogData)

# Display rates
FrenchDemogData$rate

# Export mortality data 
save(FrenchDemogData, file="MortalityFrance.RData")
