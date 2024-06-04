import matplotlib.pyplot as plt
import numpy as np

#left-right turns -- chair approacing left then right -- side to side, small lab, big lab, outdoors
#First set of experiments, original model
experiments1=['1','2','3','4','5','6','7','8','9','10','11','12']

indexes1=np.array([-0.6, -0.875, 1, 0.55, 0.25, 0.89, 0.78, 0.78, 0.81, 0.45, 0.48, 0.5])
variation1=np.array([0.1, 0.125, 0, 0.22, 0.22, 0.11, 0.22, 0.22, 0.19, 0.18, 0.22, 0.21])

fig1, ax1=plt.subplots()
ax1.bar(experiments1, indexes1, yerr=variation1, ecolor='black', capsize=5)
#ax.fill_between(experiments, indexes - variation, indexes + variation, alpha=0.2, color='gray', label='Varianza')
ax1.set_label('Experiments')
ax1.set_ylabel('Indexes')
ax1.set_title('Index with variation in different experiments')

#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#Second set of experiments
#History object using images 2 times
#Single obstacle
experiments2=['1','2','3','4'] #no bis-- history empty /bis--filled
indexes2=np.array([-0.75, -0.75, -0.75, -1])
variation2=np.array([0, 0, 0, 0,])

fig2, ax2=plt.subplots()
bar1=ax2.bar(experiments2[0], indexes2[0], yerr=variation2[0], ecolor='black', capsize=5, color='blue')
bar2=ax2.bar(experiments2[1], indexes2[1], yerr=variation2[1], ecolor='black', capsize=5, color='purple')
bar3=ax2.bar(experiments2[2], indexes2[2], yerr=variation2[2], ecolor='black', capsize=5, color='blue')
bar4=ax2.bar(experiments2[3], indexes2[3], yerr=variation2[3], ecolor='black', capsize=5, color='purple')
#ax2.bar(experiments2, indexes2, yerr=variation2, ecolor='r', capsize=5, color=['blue', 'purple', 'blue', 'purple'])
#ax.fill_between(experiments, indexes - variation, indexes + variation, alpha=0.2, color='gray', label='Varianza')
ax2.set_xlabel('Experiments')
ax2.set_ylabel('Indexes')
ax2.set_title('Index with variation in single obstacle experiments')
plt.xticks(rotation=0)
plt.legend(['history empty','history filled'])
ax2.text(0.5, -0.9, 'Static chair at right', ha='center', va='center')
ax2.text(2.5, -1.2, 'Static chair at left', ha='center', va='center')

ax2.set_ylim(-1.3,0)


#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#Two obstacle setups
#Left turns -right turns
#nobis--empty/bis--full
experiments3=['3', '3 bis','4', '4 bis','5', '5 bis','6', '6 bis','7', '7 bis','8', '8 bis']
indexes3=np.array([0.5, 0.75, 0.5, 0,  -0.25, -0.5, 1, 1, 1, 1, 1, 1])
variation3=np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0, 0, 0, 0, 0, 0])

fig3, ax3=plt.subplots()
#ax3.bar(experiments3, indexes3, yerr=variation3, ecolor='r', capsize=5)
bar1=ax3.bar(experiments3[0], indexes3[0], yerr=variation3[0], ecolor='black', capsize=5, color='blue')
bar2=ax3.bar(experiments3[1], indexes3[1], yerr=variation3[1], ecolor='black', capsize=5, color='purple')
bar3=ax3.bar(experiments3[2], indexes3[2], yerr=variation3[2], ecolor='black', capsize=5, color='blue')
bar4=ax3.bar(experiments3[3], indexes3[3], yerr=variation3[3], ecolor='black', capsize=5, color='purple')
bar5=ax3.bar(experiments3[4], indexes3[4], yerr=variation3[4], ecolor='black', capsize=5, color='blue')
bar6=ax3.bar(experiments3[5], indexes3[5], yerr=variation3[5], ecolor='black', capsize=5, color='purple')
bar7=ax3.bar(experiments3[6], indexes3[6], yerr=variation3[6], ecolor='black', capsize=5, color='blue')
bar8=ax3.bar(experiments3[7], indexes3[7], yerr=variation3[7], ecolor='black', capsize=5, color='purple')
bar9=ax3.bar(experiments3[8], indexes3[8], yerr=variation3[8], ecolor='black', capsize=5, color='blue')
bar10=ax3.bar(experiments3[9], indexes3[9], yerr=variation3[9], ecolor='black', capsize=5, color='purple')
bar11=ax3.bar(experiments3[10], indexes3[10], yerr=variation3[10], ecolor='black', capsize=5, color='blue')
bar12=ax3.bar(experiments3[11], indexes3[11], yerr=variation3[11], ecolor='black', capsize=5, color='purple')
ax3.legend(['history empty','history filled'])
#ax.fill_between(experiments, indexes - variation, indexes + variation, alpha=0.2, color='gray', label='Varianza')
ax3.set_label('Experiments')
ax3.set_ylabel('Indexes')
ax3.set_title('Index with variation in two obstacle experiments')

"""
#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#Real-life simulations
 No n'he fet perquè no era conclusiu (té sentit? ho he de fer?)
experiments=['9', '10', '11', '12']
indexes=np.array([0.7, 0.59, 0.36, 0.5])
variation=np.array([0.15, 0.05, 0.1, 0.05])

fig, ax=plt.subplots()
ax.bar(experiments, indexes, yerr=variation, ecolor='r', capsize=5)
#ax.fill_between(experiments, indexes - variation, indexes + variation, alpha=0.2, color='gray', label='Varianza')
ax.set_label('Experiments')
ax.set_ylabel('Indexes')
ax.set_title('Index with variation in real-life simulation experiments')
plt.show()
"""
#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#Second set of experiments
#History object using images 4 times
#Single obstacle
experiments5=['1','1 bis','2','2 bis'] #no bis-- history empty /bis--filled
indexes5=np.array([0.1, 0.1, -1, -1])
variation5=np.array([0.1, 0.1, 0, 0,])

fig5, ax5=plt.subplots()
bar1=ax5.bar(experiments5[0], indexes5[0], yerr=variation5[0], ecolor='black', capsize=5, color='blue')
bar1=ax5.bar(experiments5[1], indexes5[1], yerr=variation5[1], ecolor='black', capsize=5, color='purple')
bar1=ax5.bar(experiments5[2], indexes5[2], yerr=variation5[2], ecolor='black', capsize=5, color='blue')
bar1=ax5.bar(experiments5[3], indexes5[3], yerr=variation5[3], ecolor='black', capsize=5, color='purple')
ax5.legend(['hsitory empty', 'history filled'])
#ax5.bar(experiments5, indexes5, yerr=variation5, ecolor='r', capsize=5)
#ax.fill_between(experiments, indexes - variation, indexes + variation, alpha=0.2, color='gray', label='Varianza')
ax5.set_label('Experiments')
ax5.set_ylabel('Indexes')
ax5.set_title('Index with variation in single obstacle experiments')

#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#Two obstacle setups
#Left turns -right turns
#nobis--empty/bis--full
experiments6=['3', '3 bis','4', '4 bis','5', '5 bis','6', '6 bis','7', '7 bis','8', '8 bis']
indexes6=np.array([1, 1, 0.25, 0.25,  0.125, -0.75, 1, 1, 1, 1, 1, 1])
variation6=np.array([0, 0, 0.125, 0.125, 0.125, 0, 0, 0, 0, 0, 0, 0])

fig6, ax6=plt.subplots()
bar1=ax6.bar(experiments6[0], indexes6[0], yerr=variation6[0], ecolor='black', capsize=5, color='blue')
bar2=ax6.bar(experiments6[1], indexes6[1], yerr=variation6[1], ecolor='black', capsize=5, color='purple')
bar3=ax6.bar(experiments6[2], indexes6[2], yerr=variation6[2], ecolor='black', capsize=5, color='blue')
bar4=ax6.bar(experiments6[3], indexes6[3], yerr=variation6[3], ecolor='black', capsize=5, color='purple')
bar5=ax6.bar(experiments6[4], indexes6[4], yerr=variation6[4], ecolor='black', capsize=5, color='blue')
bar6=ax6.bar(experiments6[5], indexes6[5], yerr=variation6[5], ecolor='black', capsize=5, color='purple')
bar7=ax6.bar(experiments6[6], indexes6[6], yerr=variation6[6], ecolor='black', capsize=5, color='blue')
bar8=ax6.bar(experiments6[7], indexes6[7], yerr=variation6[7], ecolor='black', capsize=5, color='purple')
bar9=ax6.bar(experiments6[8], indexes6[8], yerr=variation6[8], ecolor='black', capsize=5, color='blue')
bar10=ax6.bar(experiments6[9], indexes6[9], yerr=variation6[9], ecolor='black', capsize=5, color='purple')
bar11=ax6.bar(experiments6[10], indexes6[10], yerr=variation6[10], ecolor='black', capsize=5, color='blue')
bar12=ax6.bar(experiments6[11], indexes6[11], yerr=variation6[11], ecolor='black', capsize=5, color='purple')
#ax6.bar(experiments6, indexes6, yerr=variation6, ecolor='r', capsize=5)
#ax.fill_between(experiments, indexes - variation, indexes + variation, alpha=0.2, color='gray', label='Varianza')
ax6.set_label('Experiments')
ax6.set_ylabel('Indexes')
ax6.set_title('Index with variation in two obstacle experiments')

#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#Real-life simulations
experiments7=['9', '10', '11', '12']
indexes7=np.array([0.7, 0.59, 0.36, 0.5])
variation7=np.array([0.15, 0.05, 0.1, 0.05])

fig7, ax7=plt.subplots()
ax7.bar(experiments7, indexes7, yerr=variation7, ecolor='black', capsize=5)
#ax.fill_between(experiments, indexes - variation, indexes + variation, alpha=0.2, color='gray', label='Varianza')
ax7.set_label('Experiments')
ax7.set_ylabel('Indexes')
ax7.set_title('Index with variation in real-life simulation experiments')
plt.show()
