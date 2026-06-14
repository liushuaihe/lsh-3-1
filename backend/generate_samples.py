from PIL import Image, ImageDraw
import os

output_dir = os.path.join(os.path.dirname(__file__), 'static', 'images', 'samples')
os.makedirs(output_dir, exist_ok=True)

samples = [
    ('airplane', (135, 206, 235), 'plane'),
    ('automobile', (255, 99, 71), 'car'),
    ('bird', (255, 215, 0), 'bird'),
    ('cat', (169, 169, 169), 'cat'),
    ('deer', (139, 69, 19), 'deer'),
    ('dog', (210, 180, 140), 'dog'),
    ('frog', (34, 139, 34), 'frog'),
    ('horse', (128, 128, 128), 'horse'),
    ('ship', (70, 130, 180), 'ship'),
    ('truck', (119, 136, 153), 'truck')
]

for name, color, shape in samples:
    img = Image.new('RGB', (128, 128), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    if shape == 'plane':
        draw.polygon([(64, 20), (40, 80), (88, 80)], fill=color)
        draw.rectangle([30, 70, 98, 90], fill=color)
        draw.polygon([(64, 90), (50, 110), (78, 110)], fill=(100, 100, 100))
    elif shape == 'car':
        draw.rectangle([20, 50, 108, 90], fill=color)
        draw.rectangle([35, 30, 93, 55], fill=color)
        draw.ellipse([25, 80, 45, 100], fill=(40, 40, 40))
        draw.ellipse([83, 80, 103, 100], fill=(40, 40, 40))
    elif shape == 'bird':
        draw.ellipse([40, 45, 88, 85], fill=color)
        draw.ellipse([55, 35, 80, 55], fill=color)
        draw.polygon([(80, 45), (100, 42), (80, 52)], fill=(255, 140, 0))
        draw.ellipse([68, 40, 73, 45], fill=(0, 0, 0))
        draw.ellipse([25, 50, 50, 70], fill=color)
        draw.ellipse([78, 50, 103, 70], fill=color)
    elif shape == 'cat':
        draw.ellipse([35, 45, 93, 100], fill=color)
        draw.ellipse([40, 30, 88, 65], fill=color)
        draw.polygon([(40, 35), (30, 10), (55, 28)], fill=color)
        draw.polygon([(88, 35), (98, 10), (73, 28)], fill=color)
        draw.ellipse([50, 42, 58, 50], fill=(0, 255, 0))
        draw.ellipse([70, 42, 78, 50], fill=(0, 255, 0))
        draw.ellipse([62, 44, 66, 48], fill=(0, 0, 0))
    elif shape == 'deer':
        draw.ellipse([45, 50, 83, 95], fill=color)
        draw.ellipse([52, 32, 76, 58], fill=color)
        draw.line([(55, 35), (45, 10), (50, 20)], fill=(139, 69, 19), width=3)
        draw.line([(73, 35), (83, 10), (78, 20)], fill=(139, 69, 19), width=3)
        draw.ellipse([58, 42, 63, 47], fill=(0, 0, 0))
        draw.ellipse([67, 42, 72, 47], fill=(0, 0, 0))
    elif shape == 'dog':
        draw.ellipse([32, 50, 96, 100], fill=color)
        draw.ellipse([40, 30, 88, 65], fill=color)
        draw.ellipse([35, 28, 48, 50], fill=(139, 69, 19))
        draw.ellipse([80, 28, 93, 50], fill=(139, 69, 19))
        draw.ellipse([52, 42, 59, 49], fill=(0, 0, 0))
        draw.ellipse([69, 42, 76, 49], fill=(0, 0, 0))
        draw.ellipse([58, 55, 70, 62], fill=(0, 0, 0))
    elif shape == 'frog':
        draw.ellipse([25, 45, 103, 100], fill=color)
        draw.ellipse([30, 25, 55, 50], fill=color)
        draw.ellipse([73, 25, 98, 50], fill=color)
        draw.ellipse([35, 30, 48, 43], fill=(255, 255, 255))
        draw.ellipse([80, 30, 93, 43], fill=(255, 255, 255))
        draw.ellipse([39, 34, 44, 39], fill=(0, 0, 0))
        draw.ellipse([84, 34, 89, 39], fill=(0, 0, 0))
        draw.arc([45, 60, 83, 85], 0, 180, fill=(0, 100, 0), width=3)
    elif shape == 'horse':
        draw.ellipse([25, 50, 95, 90], fill=color)
        draw.ellipse([75, 25, 105, 55], fill=color)
        draw.rectangle([28, 80, 38, 115], fill=color)
        draw.rectangle([45, 80, 55, 115], fill=color)
        draw.rectangle([65, 80, 75, 115], fill=color)
        draw.rectangle([82, 80, 92, 115], fill=color)
        draw.polygon([(25, 60), (5, 50), (15, 70)], fill=color)
        draw.ellipse([85, 35, 90, 40], fill=(0, 0, 0))
    elif shape == 'ship':
        draw.polygon([(20, 85), (108, 85), (95, 110), (33, 110)], fill=color)
        draw.rectangle([55, 35, 73, 85], fill=(255, 255, 255))
        draw.polygon([(64, 15), (64, 70), (100, 70)], fill=(255, 99, 71))
        draw.rectangle([50, 90, 78, 100], fill=(0, 0, 139))
    elif shape == 'truck':
        draw.rectangle([15, 55, 113, 95], fill=color)
        draw.rectangle([15, 30, 55, 60], fill=color)
        draw.rectangle([22, 38, 48, 52], fill=(135, 206, 235))
        draw.rectangle([65, 35, 105, 55], fill=(255, 255, 255))
        draw.ellipse([20, 85, 40, 105], fill=(40, 40, 40))
        draw.ellipse([88, 85, 108, 105], fill=(40, 40, 40))
    
    img = img.resize((64, 64), Image.LANCZOS)
    img.save(os.path.join(output_dir, f'{name}.png'), 'PNG')
    print(f'Generated: {name}.png')

print('Done! All sample images generated.')
