from transformers import pipeline

# Function to replace unsupported characters with plain ones
def clean_text_for_translation(text):
    replacements = {
        'ā': 'a',
        'ḫ': 'h',
        'ī': 'i',
        'ř': 'r',
        'š': 'sh',
        'ṣ': 'sh',
        'ū': 'u'
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text

def translate_cuneiform(text, translator, direction="Akkadian to English"):
    # Clean the text for unsupported characters
    text = clean_text_for_translation(text)
    
    # Prepare the input format for the model based on the translation direction
    if direction == "Akkadian to English":
        input_text = f"translate Akkadian to English: {text}"
    elif direction == "English to Sumerian":
        input_text = f"translate English to Sumerian: {text}"
    elif direction == "Sumerian to Akkadian":
        input_text = f"translate Sumerian to Akkadian: {text}"
    else:
        raise ValueError("Unsupported translation direction. Use 'Akkadian to English', 'English to Sumerian', or 'Sumerian to Akkadian'.")
    
    # Generate the translation
    result = translator(input_text, max_length=100, num_return_sequences=1)
    
    # Extract the translated text from the result
    translated_text = result[0]['generated_text'] if result else "Translation failed"
    
    return translated_text

def main():
    # Initialize the translation pipeline
    translator = pipeline("text2text-generation", model="praeclarum/cuneiform")
    
    # Sample Akkadian text (transcribed cuneiform)
    sample_akkadian_text = "1(disz){d}szul3-ma-nu-_sag man gal?_-u2 _man_ dan-nu _man kisz_"
    
    # Specify translation direction
    translation_direction = "Akkadian to English"
    
    # Translate the sample text
    translated_text = translate_cuneiform(sample_akkadian_text, translator, direction=translation_direction)
    
    # Print the results
    print("Sample Akkadian Text:")
    print(sample_akkadian_text)
    print("\nTranslated Text:")
    print(translated_text)

if __name__ == "__main__":
    main()
