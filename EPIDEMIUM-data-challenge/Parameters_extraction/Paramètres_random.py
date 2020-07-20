annee_avant=2004
annee_apres=2005
pays='Mexico'



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


qaux=rd.randint(0,419)
for q in range(420):
    aux=ligne_annee[q]
    ligne_annee[q]=ligne_annee_avant[q]
    mort_aux=clf.predict(ligne_annee.reshape((1,420)))
    
    if(q==qaux):
        mort=mort_aux
        col_mort=q
    ligne_annee[q]=aux
print(c2[col_mort+3],mort)

ligne_annee[col_mort]=ligne_annee_avant[col_mort]