import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import os

content_images_paths = ['/home/vealniycahko/STUDY/Code/machine_learning/cv/cv2/image1.png', '/home/vealniycahko/STUDY/Code/machine_learning/cv/cv2/image2.png', '/home/vealniycahko/STUDY/Code/machine_learning/cv/cv2/image3.png']
style_images_paths = ['/home/vealniycahko/STUDY/Code/machine_learning/cv/cv2/style1.png', '/home/vealniycahko/STUDY/Code/machine_learning/cv/cv2/style2.png', '/home/vealniycahko/STUDY/Code/machine_learning/cv/cv2/style3.png', '/home/vealniycahko/STUDY/Code/machine_learning/cv/cv2/style4.png', '/home/vealniycahko/STUDY/Code/machine_learning/cv/cv2/style5.png']

def load_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :] # тут происходит измерение
    return img

def save_img(image, filename):
    tf.keras.preprocessing.image.save_img(filename, image)

def style_transfer(content_path, style_path, output_dir):
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    
    # предтрен. модель с TensorFlow Hub
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    
    # выполняем перенос
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    
    # создание котолога аутпут
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # стилизованное img на сохранение
    output_path = os.path.join(output_dir, f'stylized_{os.path.basename(content_path)}_{os.path.basename(style_path)}')
    save_img(stylized_image[0], output_path)
    
    return output_path

def main():
    output_dir = 'cv/output'
    for content_path in content_images_paths:
        for style_path in style_images_paths:
            output_path = style_transfer(content_path, style_path, output_dir)
            print(f'Stylized image saved at {output_path}')

if __name__ == "__main__":
    main()
