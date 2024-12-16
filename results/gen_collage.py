from PIL import Image

def create_collage(images, output_path, grid_size=(2, 2), image_size=(300, 300)):
    # Open and resize images
    resized_images = [Image.open(img).resize(image_size) for img in images]

    # Create a blank canvas for the collage
    collage_width = grid_size[1] * image_size[0]
    collage_height = grid_size[0] * image_size[1]
    collage = Image.new('RGB', (collage_width, collage_height), color='white')

    # Paste images into the collage
    for index, img in enumerate(resized_images):
        row = index // grid_size[1]
        col = index % grid_size[1]
        x = col * image_size[0]
        y = row * image_size[1]
        collage.paste(img, (x, y))

    # Save the collage
    collage.save(output_path)
    print(f"Collage saved to {output_path}")

# List of images
images = ["./1.jpg", "./2.jpg", "./3.jpg", "./4.jpg"]

# Output file path
output_path = "collage.jpeg"

# Create the collage
create_collage(images, output_path)
