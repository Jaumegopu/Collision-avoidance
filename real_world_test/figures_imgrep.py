import matplotlib.pyplot as plt
import numpy as np

#left-right turns -- chair approacing left then right -- side to side, small lab, big lab, outdoors
#First set of experiments, original model
experiments1=['1','1"','2','2"']

indexes1=np.array([-0.6, -0.6, -0.875, -0.75])
variation1=np.array([0.1, 0.2, 0.125, 0.125])

fig1, ax1=plt.subplots()
#ax1.bar(experiments1, indexes1, yerr=variation1, ecolor='black', capsize=5)
bar1=ax1.bar(experiments1[0], indexes1[0], yerr=variation1[0], ecolor='black', capsize=5, color='blue')
bar2=ax1.bar(experiments1[1], indexes1[1], yerr=variation1[1], ecolor='black', capsize=5, color='purple')
bar3=ax1.bar(experiments1[2], indexes1[2], yerr=variation1[2], ecolor='black', capsize=5, color='blue')
bar4=ax1.bar(experiments1[3], indexes1[3], yerr=variation1[3], ecolor='black', capsize=5, color='purple')
#ax.fill_between(experiments, indexes - variation, indexes + variation, alpha=0.2, color='gray', label='Varianza')
ax1.set_label('Experiments')
ax1.set_ylabel('Indexes')
ax1.set_title('Index with variation in different experiments')
plt.legend(['history empty','history filled'])
#______________________________________________________________________________________________________________________
experiments2=['3','3"','4','4"','5','5"','6','6"','7','7"','8','8"']
indexes2=[1, 1, 0.55, 0.44 , 0.25, 0.25, 0.89, 0.89, 0.78, 0.78, 0.78, 0.78]
variation2=[0, 0, 0.22, 0.33, 0.22, 0.22, 0.11, 0.11, 0.22, 0.22, 0.22, 0.22]

fig2, ax2=plt.subplots()
bar1=ax2.bar(experiments2[0], indexes2[0], yerr=variation2[0], ecolor='black', capsize=5, color='blue')
bar2=ax2.bar(experiments2[1], indexes2[1], yerr=variation2[1], ecolor='black', capsize=5, color='purple')
bar3=ax2.bar(experiments2[2], indexes2[2], yerr=variation2[2], ecolor='black', capsize=5, color='blue')
bar4=ax2.bar(experiments2[3], indexes2[3], yerr=variation2[3], ecolor='black', capsize=5, color='purple')
bar5=ax2.bar(experiments2[4], indexes2[4], yerr=variation2[4], ecolor='black', capsize=5, color='blue')
bar6=ax2.bar(experiments2[5], indexes2[5], yerr=variation2[5], ecolor='black', capsize=5, color='purple')
bar7=ax2.bar(experiments2[6], indexes2[6], yerr=variation2[6], ecolor='black', capsize=5, color='blue')
bar8=ax2.bar(experiments2[7], indexes2[7], yerr=variation2[7], ecolor='black', capsize=5, color='purple')
bar9=ax2.bar(experiments2[8], indexes2[8], yerr=variation2[8], ecolor='black', capsize=5, color='blue')
bar10=ax2.bar(experiments2[9], indexes2[9], yerr=variation2[9], ecolor='black', capsize=5, color='purple')
bar11=ax2.bar(experiments2[10], indexes2[10], yerr=variation2[10], ecolor='black', capsize=5, color='blue')
bar12=ax2.bar(experiments2[11], indexes2[11], yerr=variation2[11], ecolor='black', capsize=5, color='purple')

ax2.set_xlabel('Experiments')
ax2.set_ylabel('Indexes')
ax2.set_ylim(0,1.2)
ax2.set_title('Index with variation in static and dynamic obstacles experiments')
plt.xticks(rotation=0)
plt.legend(['history empty','history filled'])


#______________________________________________________________________________________________________________________

experiments3=['9','10','11','12']
indexes3=[0.81, 0.45, 0.48, 0.5]
variation3=[0.19, 0.18, 0.22, 0.21]

fig3, ax3=plt.subplots()
ax3.bar(experiments3, indexes3, yerr=variation3, ecolor='black', capsize=5)
"""bar1=ax3.bar(experiments3[0], indexes3[0], yerr=variation3[0], ecolor='black', capsize=5)
bar2=ax3.bar(experiments3[1], indexes3[1], yerr=variation3[1], ecolor='black', capsize=5)
bar3=ax3.bar(experiments3[2], indexes3[2], yerr=variation3[2], ecolor='black', capsize=5)
bar4=ax3.bar(experiments3[3], indexes3[3], yerr=variation3[3], ecolor='black', capsize=5)
"""
ax3.set_xlabel('Experiments')
ax3.set_ylabel('Indexes')
ax3.set_title('Index with variation in real life simulation experiments')
plt.xticks(rotation=0)
#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#Second set of experiments
#History object using images 2 times
#Single obstacle
experiments4=['1','1"','2','2"'] #no bis-- history empty /bis--filled
indexes4=np.array([-0.75, -0.75, -0.75, -1])
variation4=np.array([0, 0, 0, 0,])

fig4, ax4=plt.subplots()
bar1=ax4.bar(experiments4[0], indexes4[0], yerr=variation4[0], ecolor='black', capsize=5, color='blue')
bar2=ax4.bar(experiments4[1], indexes4[1], yerr=variation4[1], ecolor='black', capsize=5, color='purple')
bar3=ax4.bar(experiments4[2], indexes4[2], yerr=variation4[2], ecolor='black', capsize=5, color='blue')
bar4=ax4.bar(experiments4[3], indexes4[3], yerr=variation4[3], ecolor='black', capsize=5, color='purple')
#ax2.bar(experiments2, indexes2, yerr=variation2, ecolor='r', capsize=5, color=['blue', 'purple', 'blue', 'purple'])
#ax.fill_between(experiments, indexes - variation, indexes + variation, alpha=0.2, color='gray', label='Varianza')
ax4.set_xlabel('Experiments')
ax4.set_ylabel('Indexes')
ax4.set_title('Index with variation in single obstacle experiments')
plt.xticks(rotation=0)
plt.legend(['history empty','history filled'])
#ax2.text(0.5, -0.9, 'Static chair at right', ha='center', va='center')
#ax2.text(2.5, -1.2, 'Static chair at left', ha='center', va='center')

ax4.set_ylim(-1.3,0)


#______________________________________________________________________________________________________________________
#Two obstacle setups
#Left turns -right turns
#nobis--empty/bis--full
experiments5=['3', '3"','4', '4"','5', '5"','6', '6"','7', '7"','8', '8"']
indexes5=np.array([0.5, 0.75, 0.5, 0,  -0.25, -0.5, 1, 1, 1, 1, 1, 1])
variation5=np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0, 0, 0, 0, 0, 0])

fig5, ax5=plt.subplots()
#ax3.bar(experiments3, indexes3, yerr=variation3, ecolor='r', capsize=5)
bar1=ax5.bar(experiments5[0], indexes5[0], yerr=variation5[0], ecolor='black', capsize=5, color='blue')
bar2=ax5.bar(experiments5[1], indexes5[1], yerr=variation5[1], ecolor='black', capsize=5, color='purple')
bar3=ax5.bar(experiments5[2], indexes5[2], yerr=variation5[2], ecolor='black', capsize=5, color='blue')
bar4=ax5.bar(experiments5[3], indexes5[3], yerr=variation5[3], ecolor='black', capsize=5, color='purple')
bar5=ax5.bar(experiments5[4], indexes5[4], yerr=variation5[4], ecolor='black', capsize=5, color='blue')
bar6=ax5.bar(experiments5[5], indexes5[5], yerr=variation5[5], ecolor='black', capsize=5, color='purple')
bar7=ax5.bar(experiments5[6], indexes5[6], yerr=variation5[6], ecolor='black', capsize=5, color='blue')
bar8=ax5.bar(experiments5[7], indexes5[7], yerr=variation5[7], ecolor='black', capsize=5, color='purple')
bar9=ax5.bar(experiments5[8], indexes5[8], yerr=variation5[8], ecolor='black', capsize=5, color='blue')
bar10=ax5.bar(experiments5[9], indexes5[9], yerr=variation5[9], ecolor='black', capsize=5, color='purple')
bar11=ax5.bar(experiments5[10], indexes5[10], yerr=variation5[10], ecolor='black', capsize=5, color='blue')
bar12=ax5.bar(experiments5[11], indexes5[11], yerr=variation5[11], ecolor='black', capsize=5, color='purple')
ax5.legend(['history empty','history filled'])
#ax.fill_between(experiments, indexes - variation, indexes + variation, alpha=0.2, color='gray', label='Varianza')
ax5.set_label('Experiments')
ax5.set_ylabel('Indexes')
ax5.set_title('Index with variation in two obstacle experiments')


#______________________________________________________________________________________________________________________
#Real-life simulations
experiments6=['9', '10', '11', '12']
indexes6=np.array([0.72, 0.63, 0.30, 0.55])
variation6=np.array([0.17, 0.1, 0.15, 0.05])

fig6, ax6=plt.subplots()
ax6.bar(experiments6, indexes6, yerr=variation6, ecolor='black', capsize=5)
#ax6.fill_between(experiments, indexes - variation, indexes + variation, alpha=0.2, color='gray', label='Varianza')
ax6.set_label('Experiments')
ax6.set_ylabel('Indexes')
ax6.set_title('Index with variation in real-life simulation experiments')
#total sum of 2imgrep variation is 1.07

#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#Second set of experiments
#History object using images 4 times
#Single obstacle
experiments7=['1','1"','2','2"'] #no bis-- history empty /bis--filled
indexes7=np.array([0.1, 0.1, -1, -1])
variation7=np.array([0.1, 0.1, 0, 0,])

fig7, ax7=plt.subplots()
bar1=ax7.bar(experiments7[0], indexes7[0], yerr=variation7[0], ecolor='black', capsize=5, color='blue')
bar2=ax7.bar(experiments7[1], indexes7[1], yerr=variation7[1], ecolor='black', capsize=5, color='purple')
bar3=ax7.bar(experiments7[2], indexes7[2], yerr=variation7[2], ecolor='black', capsize=5, color='blue')
bar4=ax7.bar(experiments7[3], indexes7[3], yerr=variation7[3], ecolor='black', capsize=5, color='purple')
ax7.legend(['hsitory empty', 'history filled'])
#ax5.bar(experiments5, indexes5, yerr=variation5, ecolor='r', capsize=5)
#ax.fill_between(experiments, indexes - variation, indexes + variation, alpha=0.2, color='gray', label='Varianza')
ax7.set_label('Experiments')
ax7.set_ylabel('Indexes')
ax7.set_title('Index with variation in single obstacle experiments')

#______________________________________________________________________________________________________________________
#Two obstacle setups
#Left turns -right turns
#nobis--empty/bis--full
experiments8=['3', '3"','4', '4"','5', '5"','6', '6"','7', '7"','8', '8"']
indexes8=np.array([1, 1, 0.25, 0.25,  0.125, -0.75, 1, 1, 1, 1, 1, 1])
variation8=np.array([0, 0, 0.125, 0.125, 0.125, 0, 0, 0, 0, 0, 0, 0])

fig8, ax8=plt.subplots()
bar1=ax8.bar(experiments8[0], indexes8[0], yerr=variation8[0], ecolor='black', capsize=5, color='blue')
bar2=ax8.bar(experiments8[1], indexes8[1], yerr=variation8[1], ecolor='black', capsize=5, color='purple')
bar3=ax8.bar(experiments8[2], indexes8[2], yerr=variation8[2], ecolor='black', capsize=5, color='blue')
bar4=ax8.bar(experiments8[3], indexes8[3], yerr=variation8[3], ecolor='black', capsize=5, color='purple')
bar5=ax8.bar(experiments8[4], indexes8[4], yerr=variation8[4], ecolor='black', capsize=5, color='blue')
bar6=ax8.bar(experiments8[5], indexes8[5], yerr=variation8[5], ecolor='black', capsize=5, color='purple')
bar7=ax8.bar(experiments8[6], indexes8[6], yerr=variation8[6], ecolor='black', capsize=5, color='blue')
bar8=ax8.bar(experiments8[7], indexes8[7], yerr=variation8[7], ecolor='black', capsize=5, color='purple')
bar9=ax8.bar(experiments8[8], indexes8[8], yerr=variation8[8], ecolor='black', capsize=5, color='blue')
bar10=ax8.bar(experiments8[9], indexes8[9], yerr=variation8[9], ecolor='black', capsize=5, color='purple')
bar11=ax8.bar(experiments8[10], indexes8[10], yerr=variation8[10], ecolor='black', capsize=5, color='blue')
bar12=ax8.bar(experiments8[11], indexes8[11], yerr=variation8[11], ecolor='black', capsize=5, color='purple')
#ax6.bar(experiments6, indexes6, yerr=variation6, ecolor='r', capsize=5)
#ax.fill_between(experiments, indexes - variation, indexes + variation, alpha=0.2, color='gray', label='Varianza')
ax8.set_label('Experiments')
ax8.set_ylabel('Indexes')
ax8.set_title('Index with variation in two obstacle experiments')
plt.legend(['history empty','history filled'])
#______________________________________________________________________________________________________________________
#Real-life simulations
experiments9=['9', '10', '11', '12']
indexes9=np.array([0.7, 0.59, 0.36, 0.5])
variation9=np.array([0.15, 0.05, 0.1, 0.05])

fig9, ax9=plt.subplots()
ax9.bar(experiments9, indexes9, yerr=variation9, ecolor='black', capsize=5)
#ax.fill_between(experiments, indexes - variation, indexes + variation, alpha=0.2, color='gray', label='Varianza')
ax9.set_label('Experiments')
ax9.set_ylabel('Indexes')
ax9.set_title('Index with variation in real-life simulation experiments')
plt.show()
#0.575 + 0.35
#total 4imgrep variation is 0.925