import csv
import pandas as pd
import sklearn

trans = pd.read_csv("transactions.csv", sep=";")

buchungstexte = trans["Buchungstext"].value_counts()
groupbuchung = trans.groupby("Buchungstext").count()
#print(trans[trans.Buchungstext == "Lastschrift (Einzugsermächtigung)"])





#Drop Waehrung, because it does not provide any information (every transaction is in euros)
#Drop Unnamed, because it only provides index, which pandas already does
#Drop Auftragskonto, because more than a 100 entries do not have a value. Also, there are only 2 unique values
#Drop Valutadatum, because in only one instance does Valutadatum not equal Buchungstag
trans.drop(columns=["Waehrung", "Unnamed: 0", "Auftragskonto", "Valutadatum"], inplace=True)
colNames = trans.columns.values

#for convenient analysis
trans.drop(columns=["Buchungstag", "label", "Betrag"], inplace=True)
c = sklearn.feature_extraction.text.CountVectorizer()
print(trans)

