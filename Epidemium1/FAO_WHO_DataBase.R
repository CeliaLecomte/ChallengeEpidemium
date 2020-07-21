#chargement des tables FAO disponibles via l'url http://faostat3.fao.org/download/E/*/E
ae_air <- read.csv(file="C:\\Users\\benjamin-schannes\\Desktop\\Agri-Environmental Indicators\\Environment_AirClimateChange_E_All_Data.csv", sep=",", header=TRUE,dec=".")
ae_energy <- read.csv(file="C:\\Users\\benjamin-schannes\\Desktop\\Agri-Environmental Indicators\\Environment_Energy_E_All_Data.csv", sep=",", header=TRUE,dec=".")
ae_fert <- read.csv(file="C:\\Users\\benjamin-schannes\\Desktop\\Agri-Environmental Indicators\\Environment_Fertilizers_E_All_Data.csv", sep=",", header=TRUE,dec=".")
ae_land <- read.csv(file="C:\\Users\\benjamin-schannes\\Desktop\\Agri-Environmental Indicators\\Environment_Land_E_All_Data.csv", sep=";", header=TRUE,dec=".")
ae_livestock <- read.csv(file="C:\\Users\\benjamin-schannes\\Desktop\\Agri-Environmental Indicators\\Environment_Livestock_E_All_Data.csv", sep=";", header=TRUE,dec=".")
ae_pest <- read.csv(file="C:\\Users\\benjamin-schannes\\Desktop\\Agri-Environmental Indicators\\Environment_Pesticides_E_All_Data.csv", sep=",", header=TRUE,dec=".")
ae_soil <- read.csv(file="C:\\Users\\benjamin-schannes\\Desktop\\Agri-Environmental Indicators\\Environment_Soil_E_All_Data.csv", sep=",", header=TRUE,dec=".")
ae_water <- read.csv(file="C:\\Users\\benjamin-schannes\\Desktop\\Agri-Environmental Indicators\\Environment_Water_E_All_Data.csv", sep=",", header=TRUE,dec=".")
ag_tot <- read.csv(file="C:\\Users\\benjamin-schannes\\Desktop\\Agri-Environmental Indicators\\Emissions_Agriculture_Agriculture_total_E_All_Data_(Norm).csv", sep=";", header=TRUE,dec=".")
aem_tot <- read.csv(file="C:\\Users\\benjamin-schannes\\Desktop\\Agri-Environmental Indicators\\Emissions_Land_Use_Land_Use_Total_E_All_Data_(Norm).csv", sep=";", header=TRUE,dec=".")
#chargement de la table OMS pour l'incidence du cancer du pancréas disponible via l'url http://apps.who.int/gho/data/?theme=main
cancer_pancreas <- read.csv(file="C:\\Users\\benjamin-schannes\\Desktop\\Agri-Environmental Indicators\\dataset_cancer.csv", sep=";", header=TRUE,dec=",")

#packages
install.packages("dplyr")
library(dplyr)
install.packages("reshape2")
library(reshape2)
install.packages("reshape")
library(reshape)

#transposition des données en vue des jointures
ae_energy_2<-reshape(ae_energy,timevar="ItemCode",idvar=c("Country","Year"),direction="wide")
ae_fert_2<-reshape(ae_fert,timevar="ItemCode",idvar=c("Country","Year"),direction="wide")
#Concat ci-dessous est la concaténation des variables ItemCode et ElementCode
ae_land_2<-reshape(ae_land,timevar=c("Concat"),idvar=c("Country","Year"),direction="wide")
ae_livestock_2<-reshape(ae_livestock,timevar="Concat",idvar=c("Country","Year"),direction="wide")
ae_soil_2<-reshape(ae_soil,timevar="ElementCode",idvar=c("Country","Year"),direction="wide")
ag_tot_2<-reshape(ag_tot,timevar="Concat",idvar=c("Country","Year"),direction="wide")
aem_tot_2<-reshape(aem_tot,timevar="Concat",idvar=c("Country","Year"),direction="wide")

#jointures (full)
ae1 <- merge(ae_air,ae_energy_2,by=c("Country","Year"),all=T)
ae2 <- merge(ae1,ae_fert_2,by=c("Country","Year"),all=T)
ae3 <- merge(ae2,ae_land_2,by=c("Country","Year"),all=T)
ae4 <- merge(ae3,ae_livestock_2,by=c("Country","Year"),all=T)
ae5 <- merge(ae4,ae_pest,by=c("Country","Year"),all=T)
ae6 <- merge(ae5,ae_soil_2,by=c("Country","Year"),all=T)
ae7 <- merge(ae6,ae_water,by=c("Country","Year"),all=T)
ae8 <- merge(ae7,ag_tot_2,by=c("Country","Year"),all=T)
ae9 <- merge(ae8,aem_tot_2,by=c("Country","Year"),all=T)

#ajout des lags des séries temporelles associées aux différents features en colonnes
Features_label <- grep("(Item|Element).\\d",names(ae9))
Features_col <- grep("Value",names(ae9))
database_label <- ae9[,c(1,2,sort(c(Features_label,Features_col)))]
database_val <- ae9[,c(1,2,Features_col)]
database_lag <-reshape(database_val,timevar="Year",idvar=c("Country"),direction="wide")

#constitution de la base d'apprentissage
training <- merge(database_lag,cancer_pancreas,by=c("Country"),all=T)

#suppression des lignes où la variable expliquée est non renseignée
training <- training[which(is.na(training[,4372])==F),]

#variable expliquée
y <- matrix(as.numeric(matrix(training[,4372],ncol=1)),ncol=1)

#retraitement des valeurs manquantes au sein de la features matrix
x <- training[,c(2:4369)]
X <- matrix(0,nrow=dim(training)[1],ncol=(dim(x)[2]))
for (i in 1:(dim(x)[2])){
     X[,i] <- matrix(as.numeric(matrix(x[,i],ncol=1)),ncol=1)
     }

#imputation de la valeur -999 aux valeurs manquantes
for (i in 1:(dim(x)[2])){
      X[which(is.na(X[,i])==T),i] = -999
      }

#traitement à part de la variable genre (facteur)
gender <- matrix(as.numeric(matrix(training[,4371],ncol=1)),ncol=1)
gender[which(is.na(gender)==T),] = 0
gender <- factor(gender)



#régression linéaire élémentaire (pour commencer par l'analyse la plus simple)
reg <- lm(y ~ X+gender)
#summary(reg)$coeff
