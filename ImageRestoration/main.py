import os
import shutil
import cv2
import glob
import matplotlib.pyplot as plt

# Set directories for input and results
upload_folder = 'BSRGAN/testsets/RealSRSet'
result_folder = 'results'

# Create directories if they don't exist
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)

# Function to load and display images
def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Visualization utility
def display(img1, img2):
    total_figs = 5
    fig = plt.figure(figsize=(total_figs*12, 14))
    
    ax1 = fig.add_subplot(1, total_figs, 1)
    plt.title('Input image', fontsize=30)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(1, total_figs, 2)
    plt.title('BSRGAN output', fontsize=30)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(1, total_figs, 3)
    plt.title('Real-ESRGAN output', fontsize=30)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(1, total_figs, 4)
    plt.title('SwinIR output', fontsize=30)
    ax4.axis('off')
    
    ax5 = fig.add_subplot(1, total_figs, 5)
    plt.title('SwinIR-Large output', fontsize=30)
    ax5.axis('off')

    ax1.imshow(img1)
    ax2.imshow(img2['BSRGAN'])
    ax3.imshow(img2['realESRGAN'])
    ax4.imshow(img2['SwinIR'])
    ax5.imshow(img2['SwinIR-L'])

# Function to process the images using models
def process_images(upload_folder, result_folder):
    # Process using BSRGAN
    os.system('cd BSRGAN && python main_test_bsrgan.py && cd ..')
    shutil.move('BSRGAN/testsets/RealSRSet_results_x4', os.path.join(result_folder, 'BSRGAN'))

    # Process using Real-ESRGAN
    os.system('python Real-ESRGAN/inference_realesrgan.py -n RealESRGAN_x4plus --input BSRGAN/testsets/RealSRSet -s 4 --output results/realESRGAN')

    # Process using SwinIR (Normal model)
    os.system('python SwinIR/main_test_swinir.py --task real_sr --model_path experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq BSRGAN/testsets/RealSRSet --scale 4')
    shutil.move('results/swinir_real_sr_x4', 'results/SwinIR')

    # Process using SwinIR-Large
    os.system('python SwinIR/main_test_swinir.py --task real_sr --model_path experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth --folder_lq BSRGAN/testsets/RealSRSet --scale 4 --large_model')
    shutil.move('results/swinir_real_sr_x4_large', 'results/SwinIR_large')

    # Rename files in SwinIR_large
    for path in sorted(glob.glob(os.path.join('results/SwinIR_large', '*.png'))):
        os.rename(path, path.replace('SwinIR.png', 'SwinIR_large.png'))

# Display and compare results
def visualize_results(upload_folder, result_folder):
    input_list = sorted(glob.glob(os.path.join(upload_folder, '*')))
    output_list = sorted(glob.glob(os.path.join(result_folder, 'SwinIR', '*')))
    
    for input_path, output_path in zip(input_list, output_list):
        img_input = imread(input_path)
        img_output = {
            'BSRGAN': imread(output_path.replace('SwinIR', 'BSRGAN')),
            'realESRGAN': imread(output_path.replace('SwinIR', 'realESRGAN').replace('_SwinIR.png', '_realESRGAN.png')),
            'SwinIR': imread(output_path),
            'SwinIR-L': imread(output_path.replace('SwinIR', 'SwinIR_large').replace('SwinIR.png', 'SwinIR_large.png'))
        }
        display(img_input, img_output)

# Main execution
if __name__ == '__main__':
    # Input: Path to your image(s)
    image_paths = input("Enter the paths to the image files (comma-separated): ").split(',')
    
    # Move the images to upload folder
    for image_path in image_paths:
        shutil.copy(image_path.strip(), upload_folder)
    
    # Process the images
    process_images(upload_folder, result_folder)

    # Visualize the results
    visualize_results(upload_folder, result_folder)
