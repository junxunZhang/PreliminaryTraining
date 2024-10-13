import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定義一個函數進行傅立葉變換並回傳頻譜圖像
def fourier_transform(image):
    # 將圖片轉換為灰階
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 進行2D快速傅立葉變換
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    
    # 計算頻譜
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    
    return magnitude_spectrum

# 讀取兩張圖片
image1 = cv2.imread('cropped_fixed_IMG_5934u.JPG')
image2 = cv2.imread('cropped_fixed_IMG_5977.JPG')

# 進行傅立葉變換並取得頻譜圖
spectrum1 = fourier_transform(image1)
spectrum2 = fourier_transform(image2)

# 在同一個視窗中顯示原圖與頻譜圖
plt.figure(figsize=(10, 5))

# 顯示第一張圖片及其頻譜圖
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title('Image 1')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(spectrum1, cmap='gray')
plt.title('Fourier Transform of Image 1')
plt.axis('off')

# 顯示第二張圖片及其頻譜圖
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title('Image 2')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(spectrum2, cmap='gray')
plt.title('Fourier Transform of Image 2')
plt.axis('off')

# 顯示圖像
plt.tight_layout()
plt.show()
