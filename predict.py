import argparse
import json
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn.functional as F


# simple example
# python train.py "flowers"
# complete example
# python train.py "flowers" --save_dir "train_checkpoint.pth" --arch "vgg16" --learning_rate 0.002 --hidden_units 512  --epochs 2 --gpu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar='image_path', type=str, default='flowers/test/58/image_02663.jpg')
    parser.add_argument('checkpoint', metavar='checkpoint', type=str, default='train_checkpoint.pth')
    parser.add_argument('--top_k', action='store', dest="top_k", type=int, default=5)
    parser.add_argument('--category_names', action='store', dest='category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=False)
    return parser.parse_args()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    _model = getattr(torchvision.models, checkpoint['pretrained_model'])(pretrained=True)
    _model.input_size = checkpoint['input_size']
    _model.output_size = checkpoint['output_size']
    _model.learning_rate = checkpoint['learning_rate']
    _model.classifier = checkpoint['classifier']
    _model.epochs = checkpoint['epochs']
    _model.load_state_dict(checkpoint['state_dict'])
    _model.class_to_idx = checkpoint['class_to_idx']
    _model.optimizer = checkpoint['optimizer']
    return _model


def process_image(image):
    resize = 256
    crop_size = 224
    (width, height) = image.size

    if height > width:
        height = int(max(height * resize / width, 1))
        width = int(resize)
    else:
        width = int(max(width * resize / height, 1))
        height = int(resize)

    # resize image
    im = image.resize((width, height))
    # crop image
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = left + crop_size
    bottom = top + crop_size
    im = im.crop((left, top, right, bottom))

    # color channels
    im = np.array(im)
    im = im / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std
    im = np.transpose(im, (2, 0, 1))
    return im


def predict(image_path, model, topk):
    model.to('cuda')

    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(image)
    image = image.unsqueeze_(0)
    image = image.float()

    with torch.no_grad():
        output = model.forward(image.cuda())

    p = F.softmax(output.data, dim=1)

    top_p = np.array(p.topk(topk)[0][0])

    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [np.int(index_to_class[each]) for each in np.array(p.topk(topk)[1][0])]

    return top_p, top_classes


def main():
    args = parse_args()
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu

    model = load_checkpoint(checkpoint)

    # # TODO: Display an image along with the top 5 classes
    #
    # path = 'flowers/test/58/image_02663.jpg'
    #
    # top_p, classes = predict(path, loaded_model)
    # max_index = classes[0]
    #
    # fig = plt.figure(figsize=(6, 6))
    # axis_1 = plt.subplot2grid((15, 9), (0, 0), colspan=9, rowspan=9)
    # axis_2 = plt.subplot2grid((15, 9), (9, 2), colspan=5, rowspan=5)
    #
    # image = Image.open(path)
    # axis_1.axis('off')
    # axis_1.set_title(cat_to_name[str(max_index)])
    # axis_1.imshow(image)
    # labels = [cat_to_name[str(index)] for index in classes]
    # y_pos = np.arange(5)
    # axis_2.set_yticks(y_pos)
    # axis_2.set_yticklabels(labels)
    # axis_2.invert_yaxis()  # probabilities read top-to-bottom
    # axis_2.set_xlabel('Probability')
    # axis_2.barh(y_pos, top_p, xerr=0, align='center')
    #
    # plt.show()


if __name__ == "__main__":
    main()