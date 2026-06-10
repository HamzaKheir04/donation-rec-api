import pandas as pd
df = pd.read_csv('smart_filtered.csv')
print(df['User_ID'].sample(5, random_state=42).values)