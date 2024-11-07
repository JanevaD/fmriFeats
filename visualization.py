# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:04:10 2024

@author: danie
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
 
#%%
results = pd.read_csv('C:\\Users\\danie\\phd\\pharmo-fmri\\results 30.0\\Results k=2.csv')
#%%
results.set_index('nident', inplace = True)

#%%
cluster_0 = results[results['Clusters']==0]
cluster_1 = results[results['Clusters']==1]
#%%
# Generate some example data (or replace these with your actual data)
np.random.seed(0)
X = cluster_0['PANSS Positive']
Y = cluster_0['PANSS Negative']
Z = cluster_0['PANSS General Psychopatology']


# Convert X and Y into a 2D array for regression
XY = np.column_stack((X, Y))

# Fit the linear regression model
model = LinearRegression()
model.fit(XY, Z)
Z_pred = model.predict(XY)

# Calculate R^2 and correlation coefficient
r2 = r2_score(Z, Z_pred)
correlation_matrix = np.corrcoef([X, Y, Z])
correlation_coeff = correlation_matrix[0, 1], correlation_matrix[0, 2], correlation_matrix[1, 2]

# Plotting the data points and regression plane
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the actual data points
ax.scatter(X, Y, Z, color='blue', label='Data Points')

# Create a mesh grid for the regression plane
x_surf = np.linspace(X.min(), X.max(), 20)
y_surf = np.linspace(Y.min(), Y.max(), 20)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = model.coef_[0] * x_surf + model.coef_[1] * y_surf + model.intercept_

# Plot the regression plane
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5, rstride=100, cstride=100)

# Labeling and legend
ax.set_xlabel('PANSS Positive')
ax.set_ylabel('PANSS Negative')
ax.set_zlabel('PANSS General Psychopatology')
ax.set_title(f"3D Regression Plot\n Cluster 0: Disease Aware\n$R^2$: {r2:.2f} | Correlations: X-Y: {correlation_coeff[0]:.2f}, X-Z: {correlation_coeff[1]:.2f}, Y-Z: {correlation_coeff[2]:.2f}")
ax.legend()

plt.show()
#%
#%%

# Generate some example data (or replace these with your actual data)
np.random.seed(0)
X = cluster_1['PANSS Positive']
Y = cluster_1['PANSS Negative']
Z = cluster_1['PANSS General Psychopatology']


# Convert X and Y into a 2D array for regression
XY = np.column_stack((X, Y))

# Fit the linear regression model
model = LinearRegression()
model.fit(XY, Z)
Z_pred = model.predict(XY)

# Calculate R^2 and correlation coefficient
r2 = r2_score(Z, Z_pred)
correlation_matrix = np.corrcoef([X, Y, Z])
correlation_coeff = correlation_matrix[0, 1], correlation_matrix[0, 2], correlation_matrix[1, 2]

# Plotting the data points and regression plane
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the actual data points
ax.scatter(X, Y, Z, color='blue', label='Data Points')

# Create a mesh grid for the regression plane
x_surf = np.linspace(X.min(), X.max(), 20)
y_surf = np.linspace(Y.min(), Y.max(), 20)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = model.coef_[0] * x_surf + model.coef_[1] * y_surf + model.intercept_

# Plot the regression plane
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5, rstride=100, cstride=100)

# Labeling and legend
ax.set_xlabel('PANSS Positive')
ax.set_ylabel('PANSS Negative')
ax.set_zlabel('PANSS General Psychopatology')
ax.set_title(f"3D Regression Plot\n Cluster 1: Disease Unware\n$R^2$: {r2:.2f} | Correlations: X-Y: {correlation_coeff[0]:.2f}, X-Z: {correlation_coeff[1]:.2f}, Y-Z: {correlation_coeff[2]:.2f}")
ax.legend()

plt.show()

#%%
cluster_0.set_index('nident', inplace = True)
cluster_0.drop('Clusters', inplace = True, axis = 1)
#%%
cluster_1.set_index('nident', inplace = True)
cluster_1.drop('Clusters', inplace = True, axis = 1)
	
#%%
mask = np.isin(feature_index, cluster_0.index) 
cluster_0_feats = features[mask]
cluster_0_feats = pd.DataFrame(cluster_0_feats)
cluster_0_feats.index = feature_index[mask]

#%%
mask = np.isin(feature_index, cluster_1.index) 
cluster_1_feats = features[mask]
cluster_1_feats = pd.DataFrame(cluster_1_feats)
cluster_1_feats.index = feature_index[mask]

#%%
mask = np.isin(cluster_0.index, cluster_0_feats.index) 
cluster_0 = cluster_0[mask]

#%%

mask = np.isin(cluster_1.index, cluster_1_feats.index) 
cluster_1 = cluster_1[mask]

#%%

full_sumd = pd.concat([cluster_0, cluster_1])
full_sumd = full_sumd.sort_index(ascending = True)
full_feats = pd.concat([cluster_0_feats, cluster_1_feats])
full_feats.columns=features_names
full_feats = full_feats.sort_index(ascending = True)
cluster_0_feats.columns = full_feats.columns
cluster_1_feats.columns = full_feats.columns

#%%
reorder = ['LH_Vis_seg_fc', 'RH_Vis_seg_fc',
           'LH_SomMot_seg_fc', 'RH_SomMot_seg_fc', 
           'LH_DorsAttn_seg_fc','RH_DorsAttn_seg_fc',
           'LH_SalVentAttn_seg_fc', 'RH_SalVentAttn_seg_fc', 
           'LH_Limbic_seg_fc', 'RH_Limbic_seg_fc',
           'LH_Cont_seg_fc', 'RH_Cont_seg_fc',
           'LH_Default_seg_fc',  'RH_Default_seg_fc',
           
           'LH_Vis_integ_fc', 'RH_Vis_integ_fc',
           'LH_SomMot_integ_fc', 'RH_SomMot_integ_fc', 
           'LH_DorsAttn_integ_fc','RH_DorsAttn_integ_fc',
           'LH_SalVentAttn_integ_fc', 'RH_SalVentAttn_integ_fc', 
           'LH_Limbic_integ_fc', 'RH_Limbic_integ_fc',
           'LH_Cont_integ_fc', 'RH_Cont_integ_fc',
           'LH_Default_integ_fc',  'RH_Default_integ_fc',        

           'Clusters'] 


#%%
correlation_results_full = pd.DataFrame(index=full_sumd.columns, columns=full_feats.columns)

# Compute correlations
for col1 in full_sumd.columns:
    for col2 in full_feats.columns:
        correlation_results_full.loc[col1, col2] = full_sumd[col1].corr(full_feats[col2])


correlation_results_full  = correlation_results_full.astype(float)

colors = [ '#FF0101','#0E0EFF']
cmap = LinearSegmentedColormap.from_list('custum_gradient', colors)
# Visualize the correlation matrix
plt.figure(figsize=(16, 6))
sns.heatmap(correlation_results_full[reorder], annot=False, cmap='bwr', center=0, vmin=-0.5, vmax=0.5)
plt.xticks(rotation = 75,  ha = 'right')
plt.title('Full dataset')
plt.show()

#%%
mask = np.abs(correlation_results_full[reorder]) > 0.2

# Plot the heatmap
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(correlation_results_full[reorder], annot=False, cmap='bwr', center=0, vmin=-0.5, vmax=0.5)


# Add annotations conditionally
for i in range(correlation_results_full[reorder].shape[0]):
    for j in range(correlation_results_full[reorder].shape[1]):
        if mask.iloc[i, j]:
            value = correlation_results_full[reorder].iloc[i, j]
            heatmap.text(j + 0.5, i + 0.5, f'{value:.2f}', ha='center', va='center', color='white')

plt.xticks(rotation=65, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize = 8)
plt.title('Full dataset')
plt.show()

#%%

cluster_0_feats['Clusters'] = np.zeros((len(cluster_0_feats),1)) 
cluster_1_feats['Clusters'] = np.ones((len(cluster_1_feats),1))
features = pd.concat([cluster_0_feats, cluster_1_feats])
features = features[reorder]
#%%
features_m = pd.melt(features, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')

#%%
fig, axes = plt.subplots(2,1, figsize = (16,8))

colors = ['#FF0101','#0E0EFF']

sns.stripplot(
    x='Feature', y='Value', hue='Clusters',
  #  kind='strip',  
    data=features_m[0:1176],
    palette=colors,
    dodge = True,
    size = 4,
    alpha = 0.6,
    ax=axes[0],
    legend = False
)

sns.boxplot(
    x='Feature', y='Value', hue='Clusters',
    data=features_m[0:1176],
    palette=colors,
    fill = False,
 #   gap = 0.5,
    dodge=True,
  #  width = 0.3,
    ax=axes[0],
    legend = False,   
)

sns.violinplot(x = 'Feature', y = 'Value', hue = 'Clusters',
               data=features_m[0:1176],
               split = True,
               inner=None,
               fill = False,
               palette=colors,
               gap = 1, 
               ax=axes[0],
               legend = False)

_ = [label.set_rotation(30) or label.set_ha('right') for label in axes[0].get_xticklabels()]

axes[0].set_title('Segregated FC by Cluster')


sns.stripplot(
    x='Feature', y='Value', hue='Clusters',
    #kind='strip',  
    data=features_m[1176:-1],
    palette=colors,
    ax=axes[1],
    size = 3,
    alpha = 0.6,
    dodge = True,
    #legend=False
)

sns.boxplot(
    x='Feature', y='Value', hue='Clusters',
    data=features_m[1176:-1],
    palette=colors,
    fill = False,
    dodge = True,
    ax = axes[1],
   # width = 0.3,
    #gap = 0.4,
    legend = False  
)

sns.violinplot(x = 'Feature', y = 'Value', hue = 'Clusters',
               data=features_m[1176:-1],
               split = True,
               inner = None, 
               fill = False, 
               gap = 1,
               palette=colors,
               ax=axes[1],
               legend = False)



_ = [label.set_rotation(30) or label.set_ha('right') for label in axes[1].get_xticklabels()]
axes[1].set_title('Integrated FC by Cluster')
plt.tight_layout()  
plt.show()

#%%
fig, axes = plt.subplots(2,1, figsize = (16,8))

colors = ['#FF0101','#0E0EFF']

sns.stripplot(
    x='Feature', y='Value', hue='Clusters',
  #  kind='strip',  
    data = features_m[0:1176],
    palette=colors,
    dodge = True,
    size = 4,
    alpha = 0.6,
    ax=axes[0],
  
)

sns.boxplot(
    x='Feature', y='Value', hue='Clusters',
    data = features_m[0:1176],
    palette=colors,
 #   gap = 0.5,
    dodge=True,
    boxprops=dict(facecolor="none") , 
  #  width = 0.3,
    ax=axes[0],
    
)



_ = [label.set_rotation(30) or label.set_ha('right') for label in axes[0].get_xticklabels()]

axes[0].set_title('Segregated FC by Cluster')


sns.stripplot(
    x='Feature', y='Value', hue='Clusters',
    #kind='strip',  
    data=features_m[1176:-1],
    palette=colors,
    ax=axes[1],
    size = 3,
    alpha = 0.6,
    dodge = True,
    #legend=False
)

sns.boxplot(
    x='Feature', y='Value', hue='Clusters',
    data=features_m[1176:-1],
    palette=colors,
    boxprops=dict(facecolor="none")  ,
    dodge = True,
    ax = axes[1],
   # width = 0.3,
    #gap = 0.4,    
)

_ = [label.set_rotation(30) or label.set_ha('right') for label in axes[1].get_xticklabels()]
axes[1].set_title('Integrated FC by Cluster')

axes[0].get_legend().remove()
axes[1].get_legend().remove()
plt.tight_layout()  
plt.show()

#%%
fig, axes = plt.subplots(2,1, figsize = (20, 10))
ptitprince.RainCloud(x= 'Feature', y = 'Value', hue = 'Clusters',
               data=features_m[1176:-1], ax=axes, dodge=True, )

#%%
g = sns.FacetGrid(features_m, col = 'Feature', height=6)
g = g.map_dataframe(pt.RainCloud, x = 'Clusters', y="Value", data = features_m[0:1176], orient = "h" )
g.fig.subplots_adjust(top = 0.75)

#%%
fig, axes = plt.subplots(2,1, figsize = (20,10))

plot_rainclouds.plot_RainClouds(x = "Feature", y = 'Value', hue = 'Clusters', data = features_m[0:1176],
                num_samples=1000, swarm_sample=False,
              order = None, hue_order = None, plot_legend=False,
              orient = "v", width_viol = .4, width_box = .60, clouds=True,
              palette = colors, bw = .2, linewidth = 1, 
              scale = "area", jitter = 1, move = 0., offset = None,
              point_size = 3, ax = axes[0], pointplot = False, 
              alpha = 0.5, dodge_violin=True, dodge_boxes=True, linecolor = 'red')

plot_rainclouds.plot_RainClouds(x = "Feature", y = 'Value', hue = 'Clusters', data = features_m[1176:-1],
                num_samples=1000, swarm_sample=False,
              order = None, hue_order = None, plot_legend=False,
              orient = "v", width_viol = .4, width_box = .60, clouds=True,
              palette = colors, bw = .2, linewidth = 1, 
              scale = "area", jitter = 1, move = 0., offset = None,
              point_size = 3, ax = axes[1], pointplot = False,
              alpha = 0.5, dodge_violin=True, dodge_boxes=True, linecolor = 'red')

_ = [label.set_rotation(30) or label.set_ha('right') for label in axes[0].get_xticklabels()]
_ = [label.set_rotation(30) or label.set_ha('right') for label in axes[1].get_xticklabels()]
axes[0].set_title('Segregated FC by Cluster')
axes[1].set_title('Integrated FC by Cluster')
axes[0].get_legend().remove()
axes[1].get_legend().remove()
plt.tight_layout()  
plt.show()

#%%
fig, axes = plt.subplots(2,1, figsize = (20,10))

PtitPrinceMod.RainCloud(x = "Feature", y = 'Value', hue = 'Clusters', data = features_m[0:1176],
                num_samples=1000, swarm_sample=False,
              order = None, hue_order = None, plot_legend=False,
              orient = "v", width_viol = .4, width_box = .60, clouds=True,
              palette = colors, bw = .2, linewidth = 1, 
              scale = "area", jitter = 1, move = 0., offset = None,
              point_size = 3, ax = axes[0], pointplot = False, 
              alpha = 0.5, dodge_violin=True, dodge_boxes=True, linecolor = 'red')

PtitPrinceMod.RainCloud(x = "Feature", y = 'Value', hue = 'Clusters', data = features_m[1176:-1],
                num_samples=1000, swarm_sample=False,
              order = None, hue_order = None, plot_legend=False,
              orient = "v", width_viol = .4, width_box = .60, clouds=True,
              palette = colors, bw = .2, linewidth = 1, 
              scale = "area", jitter = 1, move = 0., offset = None,
              point_size = 3, ax = axes[1], pointplot = False,
              alpha = 0.5, dodge_violin=True, dodge_boxes=True, linecolor = 'red')

_ = [label.set_rotation(30) or label.set_ha('right') for label in axes[0].get_xticklabels()]
_ = [label.set_rotation(30) or label.set_ha('right') for label in axes[1].get_xticklabels()]
axes[0].set_title('Segregated FC by Cluster')
axes[1].set_title('Integrated FC by Cluster')
axes[0].get_legend().remove()
axes[1].get_legend().remove()
plt.tight_layout()  
plt.show()
#%%
correlation_results_0 = pd.DataFrame(index=cluster_0.columns, columns=cluster_0_feats.columns)

# Compute correlations
for col1 in cluster_0.columns:
    for col2 in cluster_0_feats.columns:
        correlation_results_0.loc[col1, col2] = cluster_0[col1]. corr(cluster_0_feats[col2])

correlation_results_0 = correlation_results_0.astype(float)
correlation_results_0.set_index(full_sumd.columns, inplace = True)
correlation_results_0.columns = full_feats.columns


# Visualize the correlation matrix
plt.figure(figsize=(16, 6))
sns.heatmap(correlation_results_0[reorder], annot=False, cmap='bwr', center=0, vmin=-0.5, vmax=0.5)
plt.title('Cluster 0')
plt.tight_layout()
plt.show()

#%%
mask = np.abs(correlation_results_0[reorder]) > 0.2

# Plot the heatmap
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(correlation_results_0[reorder], annot=False, cmap='bwr', center=0, vmin=-0.5, vmax=0.5)


# Add annotations conditionally
for i in range(correlation_results_0[reorder].shape[0]):
    for j in range(correlation_results_0[reorder].shape[1]):
        if mask.iloc[i, j]:
            value = correlation_results_0[reorder].iloc[i, j]
            heatmap.text(j + 0.5, i + 0.5, f'{value:.2f}', ha='center', va='center', color='white')

plt.xticks(rotation=65, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize = 8)
plt.title('Cluster 0')
plt.tight_layout()
plt.show()

#%%
correlation_results_1 = pd.DataFrame(index=cluster_1.columns, columns=cluster_1_feats.columns)

# Compute correlations
for col1 in cluster_1.columns:
    for col2 in cluster_1_feats.columns:
        correlation_results_1.loc[col1, col2] = cluster_1[col1].corr(cluster_1_feats[col2])

correlation_results_1 = correlation_results_1.astype(float)
correlation_results_1.set_index(full_sumd.columns, inplace = True)
correlation_results_1.columns = full_feats.columns

# Visualize the correlation matrix
plt.figure(figsize=(16, 6))
sns.heatmap(correlation_results_1, annot=False, cmap='bwr', center=0, vmin=-0.5, vmax=0.5)
plt.title('Cluster 1')
plt.show()

#%%
mask = np.abs(correlation_results_1[reorder]) > 0.2

# Plot the heatmap
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(correlation_results_1[reorder], annot=False, cmap='bwr', center=0, vmin=-0.5, vmax=0.5)


# Add annotations conditionally
for i in range(correlation_results_1[reorder].shape[0]):
    for j in range(correlation_results_1[reorder].shape[1]):
        if mask.iloc[i, j]:
            value = correlation_results_1[reorder].iloc[i, j]
            heatmap.text(j + 0.5, i + 0.5, f'{value:.2f}', ha='center', va='center', color='white')

plt.xticks(rotation=65, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize = 8)
plt.title('Cluster 1')
plt.tight_layout()
plt.show()


#%%

diff = correlation_results_0 - correlation_results_1
diff = diff.astype(float)
plt.figure(figsize=(16, 6))
sns.heatmap(diff, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Difference Cluster 0 - Cluster 1')
plt.show()

#%%
diff = correlation_results_1 - correlation_results_0
diff = diff.astype(float)
plt.figure(figsize=(16, 6))
sns.heatmap(diff, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Difference Cluster 1 - Cluster 0')
plt.show()

#%%
for col in cluster_0_feats.columns:
    fig, axes = plt.subplots(3,3, figsize = (15,20))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]  # 
        ax.scatter( cluster_0_feats[col], cluster_0[f'SUMD{i+1}'])
        ax.set_title(f'{col} vs SUMD{i+1}')
        ax.set_ylabel(f'SUMD{i+1}')
        ax.set_xlabel(col)
    
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  
    plt.suptitle(f'Class 0 vs {col}')   
    plt.savefig(f'C:\\Users\\danie\\phd\\pharmo-fmri\\neuroimaging\\class_0_SUMD_vs_{col}_scatter.png')

#%%
for col in cluster_1_feats.columns:
    fig, axes = plt.subplots(3,3, figsize = (15,20))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]  # 
        ax.scatter(cluster_1_feats[col], cluster_1[f'SUMD{i+1}'])
        ax.set_title(f'FC{col} vs SUMD{i+1}')
        ax.set_ylabel(f'SUMD{i+1}')
        ax.set_xlabel(col)
    
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.suptitle(f'Class 1 vs {col}')  
    plt.savefig (f'C:\\Users\\danie\\phd\\pharmo-fmri\\neuroimaging\\class_1_SUMD_vs_{col}_scatter.png')       

#%%
for col in cluster_0_feats.columns:
    fig, axes = plt.subplots(3,3, figsize = (15,20))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]  # 
        sns.regplot(x= cluster_0_feats[col], y = cluster_0[f'SUMD{i+1}'], ax = ax, scatter_kws={'s':20}, line_kws={'color':'red'} )
        ax.set_title(f'{col} vs SUMD{i+1}')
        ax.set_ylabel(f'SUMD{i+1}')
        ax.set_xlabel(col)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)  
    plt.suptitle(f'Class 0 vs FC {col}')   
    plt.savefig(f'C:\\Users\\danie\\phd\\pharmo-fmri\\neuroimaging\\class_0_SUMD_vs_{col}_scatter_reg.png')


#%%
for col in cluster_1_feats.columns:
    fig, axes = plt.subplots(3,3, figsize = (15,20))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]  # 
        sns.regplot(x= cluster_1_feats[col], y = cluster_1[f'SUMD{i+1}'], ax = ax, scatter_kws={'s':20}, line_kws={'color':'red'} )
        ax.set_title(f'{col} vs SUMD{i+1}')
        ax.set_ylabel(f'SUMD{i+1}')
        ax.set_xlabel(col)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)  
    plt.suptitle(f'Class 0 vs {col}')   
    plt.savefig(f'C:\\Users\\danie\\phd\\pharmo-fmri\\neuroimaging\\class_1_SUMD_vs_{col}_scatter_reg.png')


#%%
for col in cluster_0_feats.columns:
    fig, axes = plt.subplots(3,3, figsize = (15,20))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]  # 
        sns.kdeplot(x = cluster_0_feats[col],  y=cluster_0[f'SUMD{i+1}'], ax = ax, fill = True)
        ax.set_title(f'kde of {col}')
        ax.set_ylabel(f'Distribution')
        ax.set_xlabel(col)
        
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  
    plt.suptitle(f'Kde of {col}')  
    plt.savefig(f'C:\\Users\\danie\\phd\\pharmo-fmri\\neuroimaging\\class_0_SUMD_vs_{col}_kde.png')

#%%
for col in cluster_1_feats.columns:
    fig, axes = plt.subplots(3,3, figsize = (15,20))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]  # 
        sns.kdeplot(x=cluster_1_feats[col], y=cluster_1[f'SUMD{i+1}'], ax = ax, fill = True)
        ax.set_title(f'kde of {col}')
        ax.set_ylabel(f'Distribution')
        ax.set_xlabel(col)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)  
    plt.suptitle(f'Kde of {col}')  
    plt.savefig (f'C:\\Users\\danie\\phd\\pharmo-fmri\\neuroimaging\\class_1_SUMD_vs_{col}_kde.png')       
    
#%%
for col in full_feats.columns:
    fig, axes = plt.subplots(3,3, figsize = (15,20))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]  # 
        ax.scatter( full_feats[col], full_sumd[f'SUMD{i+1}'])
        ax.set_title(f'{col} vs SUMD{i+1}')
        ax.set_ylabel(f'SUMD{i+1}')
        ax.set_xlabel(col)
    
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  
    plt.suptitle(f'Full {col}')   
    plt.savefig(f'C:\\Users\\danie\\phd\\pharmo-fmri\\neuroimaging\\full_SUMD_vs_{col}_scatter.png')


#%%
for col in full_feats.columns:
    fig, axes = plt.subplots(3,3, figsize = (15,20))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]  # 
        sns.regplot(x= full_feats[col], y = full_sumd[f'SUMD{i+1}'], ax = ax, scatter_kws={'s':20}, line_kws={'color':'red'} )
        ax.set_title(f'{col} vs SUMD{i+1}')
        ax.set_ylabel(f'SUMD{i+1}')
        ax.set_xlabel(col)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)  
    plt.suptitle(f'Full FC {col}')   
    plt.savefig(f'C:\\Users\\danie\\phd\\pharmo-fmri\\neuroimaging\\full_SUMD_vs_{col}_scatter_reg.png')


#%%
for col in full_feats.columns:
    fig, axes = plt.subplots(3,3, figsize = (15,20))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]  # 
        sns.kdeplot(x = full_feats[col],  y=full_sumd[f'SUMD{i+1}'], ax = ax, fill = True)
        ax.set_title(f'kde of {col}')
        ax.set_ylabel(f'Distribution')
        ax.set_xlabel(col)
        
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  
    plt.suptitle(f'Kde of {col}')  
    plt.savefig(f'C:\\Users\\danie\\phd\\pharmo-fmri\\neuroimaging\\full_SUMD_vs_{col}_kde.png')


#%%

