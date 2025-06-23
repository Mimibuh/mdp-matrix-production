import matplotlib.pyplot as plt
import tol_colors as tc

# Get the 'bright' palette from Tol's colors
bright_palette = tc.Bright

tc.set_default_colors(cset="muted")
print(tc.Muted.indigo)


# Create a list of color hex values from the named tuple (convert to string)
bright_colors = [getattr(bright_palette, color) for color in bright_palette._fields]

# Now, bright_colors is a list of strings (hex values)
# Apply the color palette globally using Matplotlib's cycler

# Example data for plotting
x = list(range(10))
y1 = [i for i in range(10)]
y2 = [2 * i for i in range(10)]
y3 = [3 * i for i in range(10)]
y4 = [4 * i for i in range(10)]
y5 = [5 * i for i in range(10)]

# Create a plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, y1, label="Line 1")
# ax.plot(x, y2, label="Line 2")
# ax.plot(x, y3, label="Line 2")
# ax.plot(x, y4, label="Line 2")
# ax.plot(x, y5, label="Line 2")

# Add labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Plot with Paul Tol's Bright Colors")
ax.legend()

plt.show()
