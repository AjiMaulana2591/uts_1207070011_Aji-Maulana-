import cv2
import numpy as np
import matplotlib.pyplot as plt
#membaca file gambar
img = cv2.imread('kucing.jpeg')
cv2.imshow('kucing.jpeg', img) #menampilkan gambar
cv2.waitKey(0)
cv2.destroyAllWindows()

#membaca file gambar
img = cv2.imread('kucing.jpeg')
print(type(img)) #menampilkan type data

#membaca file gambar
img = cv2.imread('kucing.jpeg')
plt.imshow(img) #menampilkan gambar dengan plt
plt.show()

#membaca file gambar
img = cv2.imread('kucing.jpeg')
print(img.shape) #menampilkan resolusi
print(img.size) #menampilkan ukuran data
print(img.dtype) #menampilkan type data

#membaca file gambar
img = cv2.imread('kucing.jpeg')
b,g,r = cv2.split(img) #band blue, gree, red disimpan di variabel b,g,r
b = img[...,0] #channel biru
g = img[...,1] #channel hijau
r = img[...,2] #channel merah
cv2.imshow('kucing.jpeg', b) #menampilkan channel biru
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('kucing.jpeg', g) #menampilkan channel hijau
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('kucing.jpeg', r) #menampilkan channel merah
cv2.waitKey(0)
cv2.destroyAllWindows()

#membaca file gambar
img = cv2.imread('kucing.jpeg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); #konversi BGR ke HSV
h,s,v = cv2.split(hsv) #memisahkan hsv
cv2.imshow('kucing.jpeg', h) #menampilkan band h
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('kucing.jpeg', s) #menampilkan band s
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('kucing.jpeg', v) #menampilkan band v
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(r.ravel(),256,[0,256]); #menampilkan histogram red
plt.show()
plt.hist(g.ravel(),256,[0,256]); #menampilkan histogram green
plt.show()
plt.hist(b.ravel(),256,[0,256]); #menampilkan histogram blue
plt.show()

plt.hist(h.ravel(),256,[0,256]); #menampilkan histogram h
plt.show()
plt.hist(s.ravel(),256,[0,256]); #menampilkan histogram s
plt.show()
plt.hist(v.ravel(),256,[0,256]); #menampilkan histogram v
plt.show()

#membaca file gambar
img = cv2.imread('kucing.jpeg')
img_height = img.shape[0]
img_width = img.shape[1]
img_channel = img.shape[2]
img_type = img.dtype
#melakukan penambahan brightness dengan nilai yang menjadi parameter
img_brightness = np.zeros(img.shape, dtype=np.uint8)
def brighter(nilai):
    for y in range(0, img_height):
        for x in range(0, img_width):
            red = img[y][x][0]
            green = img[y][x][1]
            blue = img[y][x][2]
            gray = (int(red) + int(green) + int(blue)) / 3
            gray += nilai
            if gray > 255:
                gray = 255
            if gray < 0:
                gray = 0
            img_brightness[y][x] = (gray, gray, gray)
plt.imshow(img)
plt.title("sebelum")
plt.show()
brighter(100)
plt.imshow(img_brightness)
plt.title("sesudah")
plt.show()

image = cv2.imread('kucing.jpeg')
#untuk mengetahui ukuran citra lebar dan tinggi
(height, width) = image.shape[:2]
RotasiMatriks = cv2.getRotationMatrix2D((width/2, height/2), -90, 0.5)
#rotasi dapat menggunakan methode wrapAffine mengambil citra asli
RotasiCitra = cv2.warpAffine(image, RotasiMatriks,(height, width))
cv2.imshow('citra rotasi', RotasiCitra)
cv2.waitKey(0)

# fungsi untuk menampilkan gambar menggunakan matplotlib
def show_image(image, title='Image', cmap_type='gray'):
    plt.figure()
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()
#fungsi untuk menghitung distribusi probabilitas (normalisasi histogram) citra 
def calculate_normalized_histogram(image):
    # hitung histogram citra dengan 256 bins (nilai intensitas piksel 0-255)
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    #hitung jumlah total piksel
    total_pixels = image.size
    #hitung distribusi probabilitas dengan rumus ni/N
    normalized_histogram = histogram / total_pixels 
    return normalized_histogram
# baca citra dari file menggunakan OpenCV
image_path = 'kucing.jpeg' # Ganti 'path to_your_image.jpg dengan path ke file gamb 28 image cv2.imread(image_path, cv2. IMREAD GRAYSCALE)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# menampilkan citra asli
show_image(image, title='Citra Asli')
#hitung distribusi probabilitas
normalized_histogram = calculate_normalized_histogram(image)
plt.figure()
plt.bar(range(256), normalized_histogram, color='black')
plt.title('Normalisasi Histogram Citra')
plt.xlabel('Intensitas Piksel')
plt.ylabel('Distribusi Probabilitas')
plt.show()




