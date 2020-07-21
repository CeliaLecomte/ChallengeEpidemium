#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Script File: plotCancerIncidences.r 
# Date of creation: 3 mars 2016
# Date of last modification: 3 mars 2016
# Author: Seraya Maouche <seraya.maouche@iscb.org>
# Project: EpidemiumDB (http://www.epidemium.cc)
# Short Description: This script provides functionalities to plot Cancer incidences
#                          in France
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Homme = data.frame(Cancer =
  factor(c("Prostate", "Poumon", "Colon_et_Rectum", "Vessie", "Rein"),
    levels = c("Prostate", "Poumon", "Colon_et_Rectum", "Vessie", "Rein")),
  Cas = c(73609, 28033, 21524, 9283, 7358))
  
Femme = data.frame(Cancer =
  factor(c("Sein", "Colon et Rectum", "Poumon", "Corps Utérin", "Lymphome non hodgkinien"),
    levels = c("Sein", "Colon et Rectum", "Poumon", "Corps Utérin", "Lymphome non hodgkinien")),
  Cas = c(54245, 19301, 12010, 6852, 5230))
  
 
xx <-  barplot(Homme$Cas, names = Homme$Cancer,
  xlab = "Type du Cancer", ylab = "Nombre de cas",
  main = "Homme", cex.axis=0.8, cex.names=0.7, col="darkblue") 

## Add text at top of bars
text(x = xx, y = Homme$Cas, label = Homme$Cas, pos=1, cex = 1, col = "white")

## Add x-axis labels 
axis(1, at=xx, labels=Homme$Cancer, tick=FALSE, las=2, line=-0.6, cex.axis=0.7)


yy <-  barplot(Femme$Cas, names = Femme$Cancer,
  xlab = "Type du Cancer", ylab = "Nombre de cas",
  main = "Femme", cex.axis=0.7, cex.names=0.7, col="darkblue") 

## Add text at top of bars
text(x = yy, y = Femme$Cas, label = Femme$Cas, pos=1, cex = 1, col = "white")

## Add x-axis labels 
axis(1, at=yy, labels=Femme$Cancer, tick=FALSE, las=2, line=-0.6, cex.axis=0.7)
