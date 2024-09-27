import matplotlib.pyplot as plt

# Data points
x = list(range(1, 10))
mu = [355.59, 339.95, 323.83, 309.32, 294.35, 340.56, 325.36, 310.05, 294.78]
m = [356.50, 354.00, 353.00, 353.00, 352.00, 342.00, 326.00, 311.00, 296.00]
sigma = [0.025392, 0.133158, 0.218228, 0.296211, 0.374450, 0.030719, 0.030146, 0.029518, 0.03489]

# Common settings for font size
font_size = 18

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(x, mu, marker='o', color='green', label='μ')
plt.plot(x, m, marker='o', color='purple', label='m')

# Adding shaded regions
plt.axvspan(0.5, 1.5, color='blue', alpha=0.1, label='Perfect')
plt.axvspan(1.5, 5.5, color='yellow', alpha=0.2, label='Hole')
plt.axvspan(5.5, 9.5, color='gray', alpha=0.2, label='Random')

# Labels and title
plt.xlabel('PMT Set', fontsize=font_size)
plt.ylabel('Values', fontsize=font_size)
plt.title('Mean and median for alpha = pi/8', fontsize=font_size)
plt.legend(fontsize=font_size)

# Adding region labels
#plt.text(1, 375, 'Perfect', horizontalalignment='center', verticalalignment='center', fontsize=font_size)
#plt.text(3.5, 375, 'Hole', horizontalalignment='center', verticalalignment='center', fontsize=font_size)
#plt.text(7.5, 375, 'Random', horizontalalignment='center', verticalalignment='center', fontsize=font_size)

plt.grid(True)
plt.xticks(ticks=x, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.savefig('m_plot_pi_over_8.pdf', format='pdf')
plt.show()

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(x, sigma, marker='o', color='orange', label='σn')

# Adding shaded regions
plt.axvspan(0.5, 1.5, color='blue', alpha=0.1, label='Perfect')
plt.axvspan(1.5, 5.5, color='yellow', alpha=0.2, label='Hole')
plt.axvspan(5.5, 9.5, color='gray', alpha=0.2, label='Random')

# Labels and title
plt.xlabel('PMT Set', fontsize=font_size)
plt.ylabel('Values', fontsize=font_size)
plt.title('Normalized stdev for alpha = pi/8', fontsize=font_size)
plt.legend(fontsize=font_size)

# Adding region labels
#plt.text(1, 0.4, 'Perfect', horizontalalignment='center', verticalalignment='center', fontsize=font_size)
#plt.text(3.5, 0.4, 'Hole', horizontalalignment='center', verticalalignment='center', fontsize=font_size)
#plt.text(7.5, 0.4, 'Random', horizontalalignment='center', verticalalignment='center', fontsize=font_size)

plt.grid(True)
plt.xticks(ticks=x, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.savefig('stdev_plot_pi_over_8.pdf', format='pdf')
plt.show()
