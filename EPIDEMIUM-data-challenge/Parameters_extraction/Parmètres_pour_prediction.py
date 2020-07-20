annee_avant=2005
annee=2006
pays='Peru'
##attention : -10


mort=ma.inf
col_mort=-1
n,l=x_prediction.shape

vrai_prediction=clf.predict(x_prediction_scaled)
for i in range(n):
    if(tab2[i,0]==pays and tab2[i,1]==annee):

        mortalite_sans_modif=vrai_prediction[i]
    if(tab2[i,0]==pays and tab2[i,1]==annee_avant):
        mortalite_objectif=vrai_prediction[i]

i=0
while(i<n and not(tab2[i,0]==pays and tab2[i,1]==annee)):
    i+=1
ligne_annee=x_prediction_scaled[i,:]
i=0
while(i<n and not(tab2[i,0]==pays and tab2[i,1]==annee_avant)):
    i+=1
ligne_annee_avant=x_prediction_scaled[i,:].copy()



for q in range(378):
    aux=ligne_annee[q]
    ligne_annee[q]=ligne_annee_avant[q]
    mort_aux=clf.predict(ligne_annee.reshape((1,378)))
    
    if(mort>mort_aux):
        mort=mort_aux
        col_mort=q
    ligne_annee[q]=aux
print(c2[col_mort+3],mort)

ligne_annee[col_mort]=ligne_annee_avant[col_mort]


