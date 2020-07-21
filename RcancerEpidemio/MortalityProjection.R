
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Script File: MortalityProjection.R  
# Date of creation: 3 mars 2016
# Date of last modification: 3 mars 2016
# Author: Seraya Maouche <seraya.maouche@iscb.org>
# Project: Epidemium (http://www.epidemium.cc)
# Short Description: This script provides functionalities to forcast mortality.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Install required package 
install.packages("demography")
install.packages("lifecontingencies")
install.packages("forecast")
install.packages("StMoMo")

# Load required packages
library(demography)
library(forecast)
library(lifecontingencies)
library("StMoMo")
library("missDeaths")
library("ROMIplot")

# Read french mortaity data from
# Country code for France: FRACNP = France civilian population , FRATNP= France total population
FrenchDemogData <-hmd.mx(country="FRACNP", username="XXXXX", password="HGHUH", label="French")
summary(FrenchDemogData)

par(mfrow=c(1,3))
plot(FrenchDemogData ,series="male",datatype="rate", main="Male rates")
plot(FrenchDemogData ,series="female",datatype="rate", main="Female rates")
plot(FrenchDemogData ,"total",datatype="rate", main="Total rates")

par(mfrow=c(1,3))
plot(FrenchDemogData,series="male",datatype="rate",
plot.type="time", main="Male rates",xlab="Years")
plot(FrenchDemogData,series="female",datatype="rate",
plot.type="time", main="Female rates",xlab="Years")
plot(FrenchDemogData,series="total",datatype="rate",
plot.type="time", main="Total rates",xlab="Years")

# Fit Lee-Carter Model (considering a maximum age = 100)

FranceMortLcaM <-lca(FrenchDemogData,series="male",max.age=100)
FranceMortLcaF <-lca(FrenchDemogData,series="female",max.age=100)
FranceMortLcaT <-lca(FrenchDemogData,series="total",max.age=100)

par(mfrow=c(1,3))
plot(FranceMortLcaT$ax, main="ax", xlab="Age",ylab="ax",type="l")
lines(x=FranceMortLcaF$age, y=FranceMortLcaF$ax, main="ax", col="red")
lines(x=FranceMortLcaM$age, y=FranceMortLcaM$ax, main="ax", col="blue")
legend("topleft" , c("Male","Female","Total"),
cex=0.8,col=c("blue","red","black"),lty=1);
plot(FranceMortLcaT$bx, main="bx", xlab="Age",ylab="bx",type="l")

lines(x=FranceMortLcaF$age, y=FranceMortLcaF$bx, main="bx", col="red")
lines(x=FranceMortLcaM$age, y=FranceMortLcaM$bx, main="bx", col="blue")
legend("topright" , c("Male","Female","Total"),
cex=0.8,col=c("blue","red","black"),lty=1);
plot(FranceMortLcaT$kt, main="kt", xlab="Year",ylab="kt",type="l")
lines(x=FranceMortLcaF$year, y=FranceMortLcaF$kt, main="kt", col="red")
lines(x=FranceMortLcaM$year, y=FranceMortLcaM$kt, main="kt", col="blue")
legend("topright" , c("Male","Female","Total"),
cex=0.8,col=c("blue","red","black"),lty=1);

# Use forecast package to project the future kts (> 110)
forecastM <-forecast(FranceMortLcaM,h=110)
forecastF <-forecast(FranceMortLcaF,h=110)
forecastT <-forecast(FranceMortLcaT,h=110)

par(mfrow=c(1,3))
plot(forecastM$kt.f,main="Male")
plot(forecastF$kt.f,main="Female",)
plot(forecastT$kt.f,main="Total")

# Pattern of past and forecasted mortality rates according to different population for
# people aged 65
ratesM <- cbind(FrenchDemogData$rate$male[1:100,],forecastM$rate$male[1:100,])
ratesF <- cbind(FrenchDemogData$rate$female[1:100,],forecastF$rate$female[1:100,])
ratesT <- cbind(FrenchDemogData$rate$total[1:100,],forecastT$rate$total[1:100,])
par(mfrow=c(1,1))
plot(seq(min(FrenchDemogData$year),max(FrenchDemogData$year)+110),ratesF[65,],
      col="red",xlab="Years",ylab="Death Rates",type="l")
lines(seq(min(FrenchDemogData$year),max(FrenchDemogData$year)+110),ratesM[65,],
     col="blue",xlab="Years",ylab="Death Rates")
lines(seq(min(FrenchDemogData$year),max(FrenchDemogData$year)+110),ratesT[65,],
     col="black",xlab="Years",ylab="Death Rates")
legend("topright" , c("Male","Female","Total"),
      cex=0.8,col=c("blue","red","black"),lty=1);
      
      










