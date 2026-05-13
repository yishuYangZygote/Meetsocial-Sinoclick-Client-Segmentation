import pandas as pd
input_filename = 'df_k3_kmeans_all_raw.csv'
output_filename = 'df_k3_kmeans_all_raw_labeled.csv'
df = pd.read_csv(input_filename)
df.to_csv(output_filename, index=False, encoding='utf-8-sig')
# =======================================================
input_filename_2 = 'df_k4_kmeans_all_raw.csv'
output_filename_2 = 'df_k4_kmeans_all_raw_labeled.csv'
df = pd.read_csv(input_filename_2)
df.to_csv(output_filename_2, index=False, encoding='utf-8-sig')