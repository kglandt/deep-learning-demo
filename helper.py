import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2                
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

## Dog Breed Classifier helper methods ##
def get_class_names():
    
    training_transforms = transforms.Compose([transforms.Pad(38),
                                transforms.RandomRotation(15),
                                transforms.Resize((150, 150)),
                                transforms.CenterCrop((112, 112)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_set = datasets.ImageFolder('./data/dogImages/train', transform=training_transforms)
    class_names = [item[4:].replace("_", " ") for item in train_set.classes]
    return class_names

def create_dog_breed_classifier():
    # define VGG16 model
    model_transfer = models.vgg16(pretrained=True)
    for param in model_transfer.features.parameters():
        param.requires_grad = False

    dog_breed_classifier = nn.Linear(4096, 133)

    model_transfer.classifier[6] = dog_breed_classifier

    # check if CUDA is available
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        model_transfer = model_transfer.cuda()
        
    model_transfer.load_state_dict(torch.load('models/dog_model.pt'))
    
    return model_transfer


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0




def predict_breed_transfer(img_path, model, class_names):
    
    model.eval()
    # load the image and return the predicted breed
    transform = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor()])
    
    image = Image.open(img_path)
    image = transform(image)
    image = image.unsqueeze(0)
    
    if torch.cuda.is_available:
        image = image.cuda()
        
    output = F.log_softmax(model.forward(image), dim=1)
    
    probabilities = torch.exp(output)
    
    top_k_probs, top_k_classes = probabilities.topk(2, dim=1)
    
    names = []
    if torch.abs(top_k_probs[0][0] - top_k_probs[0][1]) <= .075:
        names.append(class_names[top_k_classes[0][0]])
        names.append(class_names[top_k_classes[0][1]])
        return names, top_k_probs
    names.append(class_names[top_k_classes[0][0]])
    return names, top_k_probs[0][0]


def run_app(img_path, model, class_names):
    ## handle cases for a human face, dog, and neither
    
    image = Image.open(img_path)
    
    if face_detector(img_path):
        print('Hello, human!')
        prediction = predict_breed_transfer(img_path, model, class_names)[0][0]
        plt.imshow(image)
        plt.show()
        
        image = Image.open('assets/maltese.jpg')
        plt.imshow(image)
        plt.show()
        print('You look like a {}.'.format(prediction))

    else:
        predictions, probabilities = predict_breed_transfer(img_path, model, class_names)
        
        plt.imshow(image)
        plt.show()
        
        if len(predictions) == 1:
            print('This looks like a {} to me.'.format(predictions[0]))
        else:
            print('This animal may be a mix of {} and {}. The probabilities are within {:.1f}% of eachother'\
             .format(predictions[0], predictions[0],\
                     torch.abs(probabilities[0][0] - probabilites[0][1]) * 100.0))
##########################################################################################

## Gan helper methods
from Generator import Generator
import numpy as np
import pickle as pkl

def create_gan():
    
    g_input_size = 112
    g_hidden_dimension = 32
    g_output_size = 28 * 28
    G = Generator(g_input_size, g_hidden_dimension, g_output_size)
    
    G.eval()
    return G


def generate_latent_vector():
    latent_z = np.random.uniform(low=-1, high=1, size=(10, 112))
    latent_z = torch.from_numpy(latent_z).float()
    
    return latent_z


def view_saved_samples(file, epoch):
    with open(file, 'rb') as f:
        samples = pkl.load(f)
        
    # -1 indicates final epoch's samples (the last in the list)
    view_samples(epoch, samples)
    
    
def view_saved_face_samples(file, epoch):
    with open(file, 'rb') as f:
        samples = pkl.load(f)
        
        view_face_samples(epoch, samples)

def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=2, ncols=5, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        img = img.cpu().numpy()
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        

def view_face_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32,32,3)))
        
def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
