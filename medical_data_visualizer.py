import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
bmi_calc = df['weight'] / ((df['height'] / 100.0) ** 2)
df['overweight'] = bmi_calc.apply(lambda x: 1 if x > 25 else 0)

# 3
def normalize_data(value):
    if value > 1:
        return 1
    else:
        return 0

df['cholesterol'] = df['cholesterol'].apply(normalize_data)
df['gluc'] = df['gluc'].apply(normalize_data)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')    

    # 7
    g = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    )

    # 8
    fig = g.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    pressure_mask = (df['ap_lo'] <= df['ap_hi'])
    height = df['height']
    weight = df['weight']
    height_mask = (height >= height.quantile(0.025)) & (height <= height.quantile(0.975))
    weight_mask = (weight >= weight.quantile(0.025)) & (weight <= weight.quantile(0.975))

    clean_mask = pressure_mask & height_mask & weight_mask
    
    df_heat = df[clean_mask]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.ones_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = False
    mask = ~mask

    # 14
    fig, ax = plt.subplots(figsize=(14, 10))

    # 15
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        center=0,
        fmt=".1f",
        linewidths=.6,
        vmin=-0.1,
        vmax=0.25,
        cmap='icefire',
        cbar_kws={
            "shrink": 0.6
        },
        annot_kws={"size": 8},
        ax=ax
    )

    # 16
    fig.savefig('heatmap.png')
    return fig