#################################################################################################
#                    A-R-O-U-N-D trial project
#                     An EPIDEMIUM project
#                        Challenge 1-2
#          
#    Analyses done by: Elie Toledano, Aurélie Bardet
#    Date of Analysis: January 2018
#    Datasets: From score santé and E-cancer clinical trials registry


# Scope of the anylyses: get all csv files into R to perform data tidying & wrangling before 
#   analysis.
#
#################################################################################################

#Load require libraries
require(readr)
require(dplyr)
require(tidyr)
require(stringr)

#Load the csv file downloaded after search on e-cancer website _ delete the x10 coulumn (error) due to csv misformating
INCasearchresult <- read_csv("./INCa trials 05-01_2018 - Fichier source.csv")
INCasearchresult$X10 <- NULL



#extract NCT number
INCasearchresult <- INCasearchresult %>% mutate(CTgov = str_extract(INCasearchresult$`Autres identifiants`, 'NCT[0-9]{8}'))

##Extract pathology type according to 5 classes "Lung" "Colon, rectum" "Pancréas" "Sein" "Other"
  #strategy to search for pattern did not worked as expected. Workaround has been choosen
  # I have search in INCa databases the various pathologies and downloaded results each file have then a subselection of all studies and can be later merged!
#Poumon
IncasearchresultsPoumon <- read_csv("./Incasearchresults15Jan2018-Appareilsrespiratoires.csv", col_types = cols_only(`Num enregistrement` = col_guess()))
IncasearchresultsPoumon$Poumon <- c("Poumon")

#Colon, rectum
IncasearchresultsColorectal <- read_csv("./Incasearchresults15Jan2018-Colonrectum.csv", col_types = cols_only(`Num enregistrement` = col_guess()))
IncasearchresultsColorectal$Colorectal <- c("Colorectal")

#Pancréas
IncasearchresultsPancreas <- read_csv("./Incasearchresults15Jan2018-Pancreas.csv", col_types = cols_only(`Num enregistrement` = col_guess()))

IncasearchresultsPancreas$Pancreas <- c("Pancreas")

#Sein
IncasearchresultsSein <- read_csv("./Incasearchresults15Jan2018-Sein.csv", col_types = cols_only(`Num enregistrement` = col_guess()))
IncasearchresultsSein$Sein <- c("Sein")

#Merge into main file
INCasearchresult <- INCasearchresult %>% left_join(IncasearchresultsPoumon)
INCasearchresult <- INCasearchresult %>% left_join(IncasearchresultsColorectal)
INCasearchresult <- INCasearchresult %>% left_join(IncasearchresultsPancreas)
INCasearchresult <- INCasearchresult %>% left_join(IncasearchresultsSein)


#Some studies are for all four types of cancer, I then search for studies with 4 NAs and create a column with "Others"
INCasearchresult <- INCasearchresult %>% mutate(Others = apply(INCasearchresult[11:14], 1, function(x) ifelse(sum(is.na(x)) == 4,"Others", NA))
                                                  )


#Load additional details scraped from the internet
INCaStudyDetails <-read_csv("./INCa trials 05-01_2018 - study add details.csv")



#Get Phase number from study detail to match downloaded information
INCaStudyDetails$Phase <- gsub("\n","",INCaStudyDetails$Phase)
INCaStudyDetails$Phase <- gsub("Phase","",INCaStudyDetails$Phase)
INCaStudyDetails$Phase <- gsub(" ","",INCaStudyDetails$Phase)

#Simplify Study type
INCaStudyDetails$`Study type` <- gsub("\n","",INCaStudyDetails$`Study type`)
INCaStudyDetails$`Study type` <- gsub("Étendue d'investigation","",INCaStudyDetails$`Study type`)
INCaStudyDetails$`Study type` <- gsub(" ","",INCaStudyDetails$`Study type`)
INCaStudyDetails <- INCaStudyDetails %>% separate(`Study type`, c('Mono-Multi','Geo'),sep = '-')

#Simplify Ouverture column to get the opening date of the study, to get the number of patients included in France

INCaStudyDetails$ouverture <- gsub("\n","",INCaStudyDetails$ouverture)
INCaStudyDetails$ouverture <- gsub("  ","",INCaStudyDetails$ouverture)
INCaStudyDetails$ouverture <- gsub("-","",INCaStudyDetails$ouverture)
INCaStudyDetails$ouverture <- gsub("AvancementOuverture effective le : ","Ouverture",INCaStudyDetails$ouverture)
INCaStudyDetails$ouverture <- gsub("Nombre d'inclusions prévues en France : ","--NbinclFr",INCaStudyDetails$ouverture)
INCaStudyDetails$ouverture <- gsub("Nombre d'inclusions prévues : ","--NbinclFr",INCaStudyDetails$ouverture)

INCaStudyDetails <- INCaStudyDetails %>% separate(ouverture, c('Start','ouverture2'),sep = '--')
INCaStudyDetails$Start <- gsub("Ouverture","",INCaStudyDetails$Start)

INCaStudyDetails$ouverture2 <- gsub("Nombre","--Nbincl",INCaStudyDetails$ouverture2)
INCaStudyDetails <- INCaStudyDetails %>% separate(ouverture2, c('InclinFrance','ouverture3'),sep = '--')
INCaStudyDetails$InclinFrance <- gsub("NbinclFr","",INCaStudyDetails$InclinFrance)

INCaStudyDetails$InclinFrance <- as.numeric(INCaStudyDetails$InclinFrance)
INCaStudyDetails$Start <- as.Date(INCaStudyDetails$Start, "%d/%m/%Y")

#Drop not needed columns from Study details
INCaStudyDetails$`URL étude` <- NULL
INCaStudyDetails$`URL Sites` <- NULL
INCaStudyDetails$ouverture3 <- NULL
INCaStudyDetails$X4 <- NULL
colnames(INCaStudyDetails) <- c("Num enregistrement", "Mono-Multi", "Geo", "Start", "InclinFrance", "Phase")


#Check that ref from input and from site scrapper is the same after removing parsing errors



#Load all the centers from scrapping process
Centerfromscrapping <- read_csv("./INCa trials 05-01_2018 - centres.csv")

### THe "Ville" column contains a center number (to be dropped) the city name and the department number (two digits) between brackets
Centerfromscrapping$Ville <- gsub("[0-9]{1,3}\n","", Centerfromscrapping$Ville)

Centerfromscrapping <- Centerfromscrapping %>% separate(Ville, c('Ville','Department'), sep = -6)
Centerfromscrapping$Department <- gsub(")\n","", Centerfromscrapping$Department)
Centerfromscrapping <- Centerfromscrapping %>% separate(Department, c('trash','Department'), sep = -3)
Centerfromscrapping$trash <- NULL
Centerfromscrapping$Department <- as.numeric(Centerfromscrapping$Department)

Centerfromscrapping$Department <- gsub(".\(","", Centerfromscrapping$Department, perl = TRUE)
Centerfromscrapping$Department <- as.numeric(Centerfromscrapping$Department)


Centerfromscrapping$Ville <- gsub("\n","", Centerfromscrapping$Ville)
Centerfromscrapping$Ville <- gsub("(^\\s*)","", Centerfromscrapping$Ville, perl = TRUE)
Centerfromscrapping$Ville <- gsub("(\\s*$)","", Centerfromscrapping$Ville, perl = TRUE)

######Start concatening data

INCaStudies <- full_join(INCasearchresult, INCaStudyDetails)

colnames(Centerfromscrapping) <- c("Num enregistrement", "Centre",  "Ville", "Department",  "Lat", "Lng", "Center number")

DatabaseStudies <- full_join(Centerfromscrapping, INCaStudies)

DatabaseStudies %>% group_by(Phase) %>% summarise(count=n())
tail(DatabaseStudies %>% mutate(year = format(Start, "%Y")) %>% group_by(year) %>% summarise(count=n()), 20)

rm(INCasearchresult)
rm(IncasearchresultsColorectal)
rm(IncasearchresultsPancreas)
rm(IncasearchresultsPoumon)
rm(IncasearchresultsSein)
rm(INCaStudyDetails)
rm(INCaStudies)
rm(Centerfromscrapping)
