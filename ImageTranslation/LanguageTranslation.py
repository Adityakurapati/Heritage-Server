import os
from PIL import Image
import pytesseract
from transformers import pipeline

# You may need to set the path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Uncomment and adjust if necessary

def extract_text_from_image(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Use Tesseract to do OCR on the image
        text = pytesseract.image_to_string(img, lang='eng+sum')  # Assuming 'sum' is the Tesseract language code for Sumerian
    return text.strip()

def translate_cuneiform(text, translator):
    # Prepare the input format as shown in the example
    input_text = f"translate Akkadian to English: {text}"
    
    # Generate the translation
    result = translator(input_text, max_length=100, num_return_sequences=1)
    
    # Extract the translated text from the result
    translated_text = result[0]['generated_text'] if result else "Translation failed"
    
    return translated_text

def main():
    # Initialize the translation pipeline
    translator = pipeline("text2text-generation", model="praeclarum/cuneiform")
    
    # Get the parent directory of the current file
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the image file containing cuneiform text (in the parent directory)
    cuneiform_image = os.path.join(parent_dir, "sumerian.jpg")
    
    if not os.path.exists(cuneiform_image):
        print(f"Error: File '{cuneiform_image}' not found in the current directory.")
        return
    
    # Extract text from the image
    cuneiform_text = extract_text_from_image(cuneiform_image)
    
    if not cuneiform_text:
        print("Error: No text could be extracted from the image.")
        return
    
    # Translate the text
    translated_text = translate_cuneiform(cuneiform_text, translator)
    
    # Print the results
    print("Extracted Cuneiform Text:")
    print(cuneiform_text)
    print("\nTranslated Text:")
    print(translated_text)

if __name__ == "__main__":
    main()
