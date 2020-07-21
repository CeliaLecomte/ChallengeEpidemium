# RcancerEpidemio

A series of R scripts to QC, explore, analyze, and visualize Epidemium Open Datasets (http://data.epidemium.cc/dataset).

We provide on the Epidemium Wiki R packages and resources to analyze cancer mortality and incidences data:
- Cancer Mortality page:  [Link](http://wiki.epidemium.cc/wiki/Mortalit%C3%A9_des_cancers)
- Cancer Incidences page: [Link](http://wiki.epidemium.cc/wiki/Cancer_incidences)

## Repository content:

- coreDataset2R.r:  Script to import and QC the Epidemium Core data set.
- MortalityProjection.R import mortality data from HMD and forecast mortality of french population.
- plotCancerIncidences.R: plot cancers incidences data
- HMD2R.R: Import demographic data from the Human Mortality Database (HMD)


![alt tag](https://github.com/Epidemium/R2cancerEpidemiology/blob/master/BD4Cancer_mortalityFrance.png)


Import mortality data (Epidemium Core dataset) into R

```R
datadir   <- "/home/common/data/" 
filename  <-"mortalite_1950_2013.csv"
setwd(datadir)
mortality <- read.csv(filename, header = TRUE, sep=";") 
dim(mortality) 
[1] 3191598       6 
head(mortality)    
id.cancer  sex age registry year cases.number
1        C11 Male 0-4  Albania 2009            0
2     C00-14 Male 0-4  Albania 2009            0
3 C00-15,C32 Male 0-4  Albania 2009            0
4        C15 Male 0-4  Albania 2009            0
5        C16 Male 0-4  Albania 2009            0
6     C17-21 Male 0-4  Albania 2009            0 

# To extract data for France only, you need to do a subset on the registry column.
dataFrance <- subset(mortality, mortality[,4] =="France") 
dim(dataFrance)
[1] 59850     6
```

Import cancer incidences data

```R
setwd("..BD4Cancer/data")
dat <- read.csv("incidence_1953_2007.csv", header=T, sep=",")
header(dat)

# use the function grep to extract data for France.
datFrance <- dat[grep("^France", dat$registry), ]

# Export data as .TXT file or RData Object (more efficient for further analyses)
write.table(datFrance,"incidenceCanFrance.txt", sep = "\t")
save(dat, file = "incidencesAllCoreDataset.RData")
save(datFrance, file = "incidenceCancFrance.RData")

```


