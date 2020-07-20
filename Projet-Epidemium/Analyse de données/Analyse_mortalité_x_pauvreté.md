Analyse mortalité x indicateur de pauvreté
================

Afin de pouvoir étudier l'influence des indicateurs de pauvreté sur la mortalité, on détermine d'abord les indicateurs que l'on va étudier. Dans un premier temps on regarde, le nombre de valeurs manquantes pour chacune des variables. En effet si le nombre de valeurs manquantes est trop important, il n'est pas pertinent de conserver la variable.

On crée une fonction qui calcule le nombre de valeurs manquantes pour toutes les variables :

``` r
load(file="~/PST.RData")
Poverty<-Indicators[Indicators$Category=="Poverty",]
## création du dataframe à remplir
Poverty_NA<- data.frame(matrix(ncol = 4, nrow = 0))
colnames(Poverty_NA)<-c("Code","Nom","Nombre de NA","% de NA")
for (i in 1:nrow(Poverty)) ## Poverty = dataset des indicateurs de pauvreté et leur signification
  {
      ### Calcul de NA
      code=Poverty[i,3]
      nbNA=sum(is.na(WorldBank[[toString(code)]]))
      Poverty_NA[i,]<-list(toString(code),toString(Poverty[i,4]),nbNA,nbNA/46*100)
}
##Affichage du résultat
library(knitr)
kable(Poverty_NA,caption="Valeurs manquantes pour les variables Pauvreté")
```

| Code           | Nom                                                                 |  Nombre de NA|   % de NA|
|:---------------|:--------------------------------------------------------------------|-------------:|---------:|
| SI.DST.02ND.20 | Income share held by second 20%                                     |            16|  34.78261|
| SI.DST.03RD.20 | Income share held by third 20%                                      |            16|  34.78261|
| SI.DST.04TH.20 | Income share held by fourth 20%                                     |            16|  34.78261|
| SI.DST.05TH.20 | Income share held by highest 20%                                    |            16|  34.78261|
| SI.DST.10TH.10 | Income share held by highest 10%                                    |            16|  34.78261|
| SI.DST.FRST.10 | Income share held by lowest 10%                                     |            16|  34.78261|
| SI.DST.FRST.20 | Income share held by lowest 20%                                     |            16|  34.78261|
| SI.POV.2DAY    | Poverty headcount ratio at $3.10 a day (2011 PPP) (% of population) |            16|  34.78261|
| SI.POV.DDAY    | Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population) |            16|  34.78261|
| SI.POV.GAP2    | Poverty gap at $3.10 a day (2011 PPP) (%)                           |            16|  34.78261|
| SI.POV.GAPS    | Poverty gap at $1.90 a day (2011 PPP) (%)                           |            16|  34.78261|
| SI.POV.GINI    | GINI index (World Bank estimate)                                    |            16|  34.78261|
| SI.POV.NAHC    | Poverty headcount ratio at national poverty lines (% of population) |            33|  71.73913|

On constate qu'il n'y a aucune variable avec plus de 70% des données.
