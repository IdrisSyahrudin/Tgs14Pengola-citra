|Nama |_Idris Syahrudin_|
| :- | :- |
|**Nim** |_312210467_|
|**Kelas** |_TI.22.A.5_|
|**Mata Kuliah**|_Pengolahan Citra_|
|**Tugas** |_Tgs Pertemuan 14_|

### Penguraian Pengolahan Citra

---

import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assign and open image from URL
url = 'https://i.pinimg.com/236x/09/9e/4b/099e4b650e504d4793a9422fc015e919.jpg'
response = requests.get(url, stream=True)

# Saving the image
with open('image.png', 'wb') as f:
    f.write(response.content)

# Read the image using OpenCV
img = cv2.imread('image.png')

# Converting the image into gray scale for faster computation.
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculating the SVD
u, s, v = np.linalg.svd(gray_image, full_matrices=False)

# Inspect shapes of the matrices
print(f'u.shape:{u.shape}, s.shape:{s.shape}, v.shape:{v.shape}')

# Calculate variance explained by each singular value
var_explained = np.round(s*2 / np.sum(s*2), decimals=6)

# Variance explained top Singular vectors
print(f'Variance Explained by Top 20 singular values:\n{var_explained[0:20]}')

# Plotting the variance explained by the top 20 singular values
sns.barplot(x=list(range(1, 21)), y=var_explained[0:20], color="dodgerblue")

plt.title('Variance Explained Graph')
plt.xlabel('Singular Vector', fontsize=16)
plt.ylabel('Variance Explained', fontsize=16)
plt.tight_layout()
plt.show()

# Plot images with different number of components
comps = [3648, 1, 5, 10, 15, 20]
plt.figure(figsize=(12, 6))

for i in range(len(comps)):
    low_rank = u[:, :comps[i]] @ np.diag(s[:comps[i]]) @ v[:comps[i], :]

    plt.subplot(2, 3, i + 1)
    plt.imshow(low_rank, cmap='gray')
    if i == 0:
        plt.title(f'Actual Image with n_components = {comps[i]}')
    else:
        plt.title(f'n_components = {comps[i]}')

plt.tight_layout()
plt.show()

### Hasil Output
---

![Screenshot (239)](https://github.com/IdrisSyahrudin/Tgs14Pengola-citra/assets/129921422/5c3aff69-df4d-447a-b664-2ac003bb75ea)

![Screenshot (240)](https://github.com/IdrisSyahrudin/Tgs14Pengola-citra/assets/129921422/70550ed6-1465-4f26-ad96-e4b6c24254b9)

![Screenshot (241)](https://github.com/IdrisSyahrudin/Tgs14Pengola-citra/assets/129921422/b40eb001-5919-4d4e-996e-f6d00ff7cc52)





