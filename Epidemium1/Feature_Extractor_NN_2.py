1.	import pandas as pd
2.	from sklearn.base import TransformerMixin
3.	 
4.	 
5.	class FeatureExtractor(TransformerMixin):
6.	    main_cols = ['country', 'gender', 'ageMin', 'ageMax', 'year']
7.	    inci_cols = [
8.	        # Other cancers mortality rate
9.	        'g_mNasopharynx (C11)', 'g_mBreast (C50)', 'g_mMesothelioma (C45)',
10.	        'g_mCorpus uteri (C54)', 'g_mLip, oral cavity, pharynx, larynx and oesophagus (C00-15,C32)',
11.	        'g_mMelanoma of skin (C43)', 'g_mMultiple myeloma (C88+C90)', 'g_mUterus (C53-55)',
12.	        'g_Brain, central ner', 'g_mKidney (C64)',
13.	 
14.	        # incidence of the cancers we are targeting
15.	        'incidence X21.0', 'incidence X21.1', 'incidence X21.2',
16.	        'incidence X21.3', 'incidence X21.4', 'incidence X21.5', 'incidence X21.6',
17.	        'incidence X22.0', 'incidence X22.1', 'incidence X22.2', 'incidence X22.3',
18.	        'incidence X22.4', 'incidence X22.5', 'incidence X22.6', 'incidence X22.7',
19.	        'incidence X22.8',
20.	        'incidence C00-96, C44']
21.	    def __init__(self):
22.	        pass
23.	    def fit(self, df, y):
24.	        return self
25.	    def transform(self, df):
26.	        df_ = df[self.main_cols + self.inci_cols].copy()
27.	        df_ = pd.get_dummies(df_, drop_first=False, columns=['country'])
28.	        df_ = pd.get_dummies(df_, drop_first=True, columns=['gender'])
29.	        return df_.values

