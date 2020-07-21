#################################################################################################
#                    A-R-O-U-N-D trial project
#                     An EPIDEMIUM project
#                        Challenge 1-2
#          
#    Analyses done by: Elie Toledano, Aurelie Bardet
#    Date of Analysis: January 2018
#    Datasets: from SCORE SANTE website
#    Aim of the program: formatting of file includind data on deaths
#
#################################################################################################

# needed packages
install.packages("plyr", dependencies=TRUE)
library(plyr) 

# import of initial file with the following manual modifications on csv file
# deletion of titles in csv file
# add of GENDER and PATHOLOGY columns
# split of columns for geographical zone in DEPARTMENT and REGION (manual
# function "convert" in csv file)
# replacement of apostrophes by nothing (to avoid error in R import)
# deletion of COLUMN "taux de variation annuel moyen"
# deletion of character strings " (ns)", "(nv)", " (+)" and "(-)" 
# (control H manual function)

#in the project, . = C:/Users/...

data_in_all <- read.table("./alldeath_stdrate_France_19902012.csv",header=TRUE,sep=";",dec=",")
data_in_breast <- read.table("./breastdeath_stdrate_France_19902012.csv",header=TRUE,sep=";",dec=",")
data_in_lung <- read.table("./lungdeath_stdrate_France_19902012.csv",header=TRUE,sep=";",dec=",")
data_in_pancreas <- read.table("./pancreasdeath_stdrate_France_19902012.csv",header=TRUE,sep=";",dec=",")
data_in_colon <- read.table("./colondeath_stdrate_France_19902012.csv",header=TRUE,sep=";",dec=",")

#definition of function
format_for_work<-function(datain,dataout)
{

# add of information on department numbers
datain <- rename(datain,c("DEPARTMENT"="DEPARTMENT_NAME"))

#add of column on department number
departments <- read.table("./departments_numbers.csv",header=TRUE,sep=";")
data_in2 <- merge(datain, departments, by = "DEPARTMENT_NAME")           

#recovery of information on years in lines (rather than in columns)
liste <- vector(mode = "list", length = 23)
for(i in 1990:2012){
  liste[[i]]<-paste("tempo", i, sep="")
  tempo<-subset(data_in2, select=c(DEPARTMENT_NAME,DEPARTMENT,
                REGION,GENDER,PATHOLOGY,get(paste("X",i,sep=""))))
  tempo<-cbind(tempo,YEAR=i)
  DEATHRATE=tempo[,6]
  tempo<-cbind(tempo,DEATHRATE)
  tempo<-tempo[,-c(6)]
  assign(paste("tempo", i, sep=""), tempo)

}

#concatenation of all years
dataout0<-rbind(tempo2002,tempo2003,tempo2004,
                tempo2005,tempo2006,tempo2007,tempo2008,tempo2009, 
                tempo2010,tempo2011,tempo2012)

#deletion of departments not in metropolitan France
dataout<-dataout0[!(dataout0$DEPARTMENT %in% c("0","971","972","973","974")),]


#creation of dataout inside function
#print(dataout)

}

#use of function with creation of database
final_all<-format_for_work(datain=data_in_all,dataout=final_all)
final_breast<-format_for_work(datain=data_in_breast,dataout=final_breast)
final_pancreas<-format_for_work(datain=data_in_pancreas,dataout=final_pancreas)
final_colon<-format_for_work(datain=data_in_colon,dataout=final_colon)
final_lung<-format_for_work(datain=data_in_lung,dataout=final_lung)

##creation of a temporary merge to calculate rate of death for "Other" cancers

#additional column with DEATHRATE to include name of the pathology to perform future merge between all
#databases to recover a deathrate for "OTHER" cancers
tpofi_all <- rename(final_all,c("DEATHRATE"="ALLDEATHRATE"))
tpofi_breast <- rename(final_breast,c("DEATHRATE"="BREASTDEATHRATE"))
tpofi_pancreas <- rename(final_pancreas,c("DEATHRATE"="PANCREASDEATHRATE"))
tpofi_colon <- rename(final_colon,c("DEATHRATE"="COLONDEATHRATE"))
tpofi_lung <- rename(final_lung,c("DEATHRATE"="LUNGDEATHRATE"))

#deletion of variable PATHOLOGY in order to avoid future warning messages during merge 
tpofi_all<-subset(tpofi_all, select=-c(PATHOLOGY))
tpofi_breast<-subset(tpofi_breast, select=-c(PATHOLOGY))
tpofi_colon<-subset(tpofi_colon, select=-c(PATHOLOGY))
tpofi_pancreas<-subset(tpofi_pancreas, select=-c(PATHOLOGY))
tpofi_lung<-subset(tpofi_lung, select=-c(PATHOLOGY))
calc_other<-merge(tpofi_all,tpofi_breast,by=c("DEPARTMENT_NAME","DEPARTMENT","REGION","GENDER","YEAR"))           
calc_other<-merge(calc_other,tpofi_pancreas,by=c("DEPARTMENT_NAME","DEPARTMENT","REGION","GENDER","YEAR"))
calc_other<-merge(calc_other,tpofi_colon,by=c("DEPARTMENT_NAME","DEPARTMENT","REGION","GENDER","YEAR"))
calc_other<-merge(calc_other,tpofi_lung,by=c("DEPARTMENT_NAME","DEPARTMENT","REGION","GENDER","YEAR"))

#check on missing values
sum(is.na(calc_other$ALLDEATHRATE))
sum(is.na(calc_other$BREASTDEATHRATE))
sum(is.na(calc_other$COLONDEATHRATE))
sum(is.na(calc_other$PANCREASDEATHRATE))
sum(is.na(calc_other$LUNGDEATHRATE))

calc_other$DEATHRATE<-calc_other$ALLDEATHRATE-calc_other$BREASTDEATHRATE-calc_other$PANCREASDEATHRATE-calc_other$COLONDEATHRATE-calc_other$LUNGDEATHRATE
calc_other$PATHOLOGY<-vector(length=length(calc_other$DEATHRATE))
calc_other$PATHOLOGY<-c("Other")
final_other<-subset(calc_other,select=-c(ALLDEATHRATE,BREASTDEATHRATE,PANCREASDEATHRATE,COLONDEATHRATE,LUNGDEATHRATE))

#gathering of information in final table    
FINAL<-rbind(final_all,final_breast,final_lung,final_colon,final_pancreas,final_other)
write.table(FINAL, file = "./final_table_for_death.csv", 
            sep = ",", row.names = TRUE, col.names = TRUE)



#end of program#
