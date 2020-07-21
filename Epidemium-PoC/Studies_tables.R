#################################################################################################
#                    A-R-O-U-N-D trial project
#                     An EPIDEMIUM project
#                        Challenge 1-2
#          
#    Analyses done by: Elie Toledano, Aurélie Bardet
#    Date of Analysis: January 2018
#    Datasets: From score santé and E-cancer clinical trials registry


# Scope of the anylyses: Restrict data to scope of analysis and produce tables for viz'
# Scope : 2002 - 2012 and look for cumulative number of studies or start per year
#################################################################################################
require(dplyr)
require(tidyr)

## Remove year not of interest
Studies <- DatabaseStudies %>% mutate(year = format(Start, "%Y")) %>% filter(year >2001 & year <2013) %>% drop_na(Department)

## Remove not relevant variables
Studies$`Num enregistrement` <-NULL
Studies$Ville <-NULL
Studies$Lat <-NULL
Studies$Lng <-NULL
Studies$`Autres identifiants` <-NULL
Studies$Coordonnateur <-NULL
Studies$Pathologies <-NULL
Studies$url <-NULL
Studies$CTgov <-NULL
Studies$`Mono-Multi` <-NULL
Studies$Geo <-NULL
Studies$`État de l'essai` <-NULL
Studies$ <-NULL

###Define dataset to be vizualized

StudiesAll <- Studies %>% group_by(Department, year) %>% summarise(count=n()) %>% group_by(Department) %>% mutate(CumCount = cumsum(count))
StudiesPoumon <- Studies %>% filter(Poumon == "Poumon") %>% group_by(Department, year) %>% summarise(count=n()) %>% group_by(Department) %>% mutate(CumCount = cumsum(count))
StudiesSein <- Studies %>% filter(Sein == "Sein") %>% group_by(Department, year) %>% summarise(count=n()) %>% group_by(Department) %>% mutate(CumCount = cumsum(count))
StudiesColorectal <- Studies %>% filter(Colorectal == "Colorectal") %>% group_by(Department, year) %>% summarise(count=n()) %>% group_by(Department) %>% mutate(CumCount = cumsum(count))
StudiesPancreas <- Studies %>% filter(Pancreas == "Pancreas") %>% group_by(Department, year) %>% summarise(count=n()) %>% group_by(Department) %>% mutate(CumCount = cumsum(count))


