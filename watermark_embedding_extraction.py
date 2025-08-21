import numpy as np
import pywt
import cv2
import pickle
import matplotlib.pyplot as plt

def arnold_cat_map(image, iterations, inverse=False):
    N = image.shape[0]
    result = np.copy(image)
    for _ in range(iterations):
        new_image = np.zeros_like(result)
        for x in range(N):
            for y in range(N):
                if not inverse:
                    new_x = (x + y) % N
                    new_y = (x + 2 * y) % N
                else:
                    new_x = (2 * x - y) % N
                    new_y = (-x + y) % N
                new_image[new_x, new_y] = result[x, y]
        result = new_image
    return result

def dwt_2d(image):
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    return LL, (LH, HL, HH)

def idwt_2d(LL, coeffs):
    return pywt.idwt2((LL, coeffs), 'haar')

def dct_2d(block):
    return cv2.dct(block.astype(np.float32))

def idct_2d(block):
    return cv2.idct(block)

def calculate_p(L, B, n=16):
    alpha = (((B[0]+B[1]) / n )+ L*n)/L
    P = L / alpha
    return P, alpha

def embedding_watermark(host_image, watermark, n=16, coordinate_file='coordinates.pkl'):
    watermark_size = watermark.shape[0] * watermark.shape[1]
    watermark_bits = watermark.flatten()
    host_image_size = host_image.shape[0] * host_image.shape[1]
    host_image_bits = host_image.flatten()
    u = 0
    u2 = 0
    height, width = host_image.shape
    watermarked_image = np.copy(host_image)

    global coordinates
    coordinates = []

    blocks = []
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = host_image[i:i+8, j:j+8]
            if block.shape[0] == 8 and block.shape[1] == 8:
                variance = np.var(block)
                blocks.append((variance, i, j))

    # Sort blocks by variance in descending order
    blocks.sort(reverse=True, key=lambda x: x[0])

    for block in blocks:
        #if u >= host_image_size:
        if u >= 262144:
            break
        variance, i, j = block
        block = host_image[i:i+8, j:j+8]

        LL, (LH, HL, HH) = dwt_2d(block)
        dct_block = dct_2d(LL)
        
        L = np.mean(dct_block)
        B = [dct_block[2, 3], dct_block[3, 2]]
        
        is_embedded = False
        if u2 < watermark_size:
            P, alpha = calculate_p(L, B, n)
            if dct_block[2, 3]<P +alpha and dct_block[3, 2] <P+ alpha  and watermark_bits[u2] == 255:
                dct_block[2, 3] = P - alpha 
                dct_block[3, 2] = P - alpha 
                #u2 += 1
                is_embedded = True
                u2 += 1
                coordinates.append((i, j, variance, u2))
                    
            elif dct_block[2, 3]>P -alpha and dct_block[3, 2] >P-alpha  and watermark_bits[u2] == 0:
                dct_block[2, 3] = P + alpha 
                dct_block[3, 2] = P + alpha
                is_embedded = True
                u2 += 1
                coordinates.append((i, j, variance, u2))
            
            else:
                dct_block[2, 3] =   dct_block[2, 3]
                dct_block[3, 2] =   dct_block[3, 2]
                is_embedded = False
            
        LL = idct_2d(dct_block)
        watermarked_block = idwt_2d(LL, (LH, HL, HH))
        watermarked_image[i:i+8, j:j+8] = watermarked_block

    with open(coordinate_file, 'wb') as f:
        pickle.dump(coordinates, f)

    return watermarked_image

def extract_watermark(watermarked_image, original_image, key, watermark_size):
    height, width = watermarked_image.shape
    watermark_bits = []

    with open('coordinates.pkl', 'rb') as f:
        coordinates = pickle.load(f)

    u1 = 0  # Initialize u1 before the loop
    for (i, j, _, _) in coordinates:
        if u1 >= watermark_size**2:
            break
        block_w = watermarked_image[i:i+8, j:j+8]
        block_o = original_image[i:i+8, j:j+8]
        if block_w.shape[0] != 8 or block_w.shape[1] != 8:
            continue

        LL_w, (LH_w, HL_w, HH_w) = dwt_2d(block_w)
        LL_o, (LH_o, HL_o, HH_o) = dwt_2d(block_o)

        dct_block_w = dct_2d(LL_w)
        dct_block_o = dct_2d(LL_o)

        B_w = np.array([dct_block_w[2, 3], dct_block_w[3, 2]])
        B_o = np.array([dct_block_o[2, 3], dct_block_o[3, 2]])

        L_o = np.mean(dct_block_o)
        L_w = np.mean(dct_block_w)
        P, alpha = calculate_p(L_w, B_w)
        
        if u1 < watermark_size**2:
            #for k in range(2):
            P, alpha = calculate_p(L_w, B_w)
            if dct_block_w[2, 3]>P and dct_block_w[3, 2]> P:
                watermark_bits.append(0)
            else:
                watermark_bits.append(255)
            u1 += 1


    print(f"Extracted watermark bits: {len(watermark_bits)}")
    print(f"Expected watermark size: {watermark_size**2}")

    if len(watermark_bits) < watermark_size**2:
        print("Warning: Not enough bits extracted to form the watermark. Padding with zeros.")
        watermark_bits.extend([0] * (watermark_size**2 - len(watermark_bits)))

    global extracted_watermark
    extracted_watermark = np.array(watermark_bits[:watermark_size**2]).reshape((watermark_size, watermark_size))
    extracted_watermark_final = arnold_cat_map(extracted_watermark, key, inverse=True)

    return extracted_watermark_final

def calculate_psnr(original_image, watermarked_image):
    mse = np.mean((original_image - watermarked_image) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr


host_image = cv2.imread("lena512512.jpg", cv2.IMREAD_GRAYSCALE)
watermark = cv2.imread("star.png", cv2.IMREAD_GRAYSCALE)
watermark_1 = arnold_cat_map(watermark, 10, inverse=False)  # Scramble the watermark

# Embed watermark
key = 10
watermarked_image = embedding_watermark(host_image, watermark_1, key)

# Save watermarked image
cv2.imwrite('watermarked_image.png', watermarked_image)

# Extract watermark
watermarked_image = cv2.imread('watermarked_image.png', cv2.IMREAD_GRAYSCALE)
recovered_watermark = extract_watermark(watermarked_image, host_image, key, watermark.shape[0])

# Plot the results
plt.figure(figsize=(15, 10))

# Plot original image
plt.subplot(2, 3, 1)
plt.title('extracted watermark Image')
plt.imshow(extracted_watermark, cmap='gray')
plt.axis('off')

# Plot original scrambled watermark
plt.subplot(2, 3, 2)
plt.title('Original Scrambled Watermark')
plt.imshow(watermark_1, cmap='gray')
plt.axis('off')

# Plot watermarked image
plt.subplot(2, 3, 3)
plt.title('Watermarked Image')
plt.imshow(watermarked_image, cmap='gray')
plt.axis('off')

# Plot recovered watermark
plt.subplot(2, 3, 4)
plt.title('Recovered Watermark')
plt.imshow(recovered_watermark, cmap='gray')
plt.axis('off')

# Plot original watermark
plt.subplot(2, 3, 5)
plt.title('Original Watermark')
plt.imshow(watermark, cmap='gray')
plt.axis('off')

# Plot original watermark
plt.subplot(2, 3, 6)
plt.title('difference image')
plt.imshow(watermarked_image-host_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Calculate and print PSNR
psnr = calculate_psnr(host_image, watermarked_image)
print(f"PSNR between original and watermarked image: {psnr} dB")

global coordinates 
print(len(coordinates))

print(watermark.size)