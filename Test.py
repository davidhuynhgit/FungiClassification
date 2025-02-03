import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, 10)
        self.relu = nn.ReLU()       
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        y = self.fc2(x)       
        return y
        
class MyClassifier():
        
    def __init__(self):
        self.class_labels = ['edible_1', 'edible_2', 'edible_3', 'edible_4', 'edible_5',
                            'poisonous_1', 'poisonous_2', 'poisonous_3', 'poisonous_4', 'poisonous_5']
                
    def setup(self):
        ''' 
            Initialise classification model. 
        '''

        imagenet_means = (0.485, 0.456, 0.406)
        imagenet_stds = (0.229, 0.224, 0.225)
        self.transform = transforms.Compose(
                        [transforms.ToTensor(), 
                         transforms.Resize((224, 224)),
                         transforms.Normalize(imagenet_means, imagenet_stds)
                        ])

        self.dino =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.dino.eval()

        self.model = DeepNet()
        self.model.load_state_dict(torch.load('DinoB_Tuned_DeepNet_balanced_lr0.001.pth', map_location = 'cpu'))
        self.model.eval()      

        
    def test_image(self, image):
        ''' 
            This function will be given a PIL image, and should return the predicted class label for that image. 
        '''
              
        transformed_input_image = self.transform(image).unsqueeze(0)
        feature = self.dino(transformed_input_image)
        output = self.model(feature)
        predicted_idx = torch.argmax(output)
        predicted_cls = self.class_labels[predicted_idx]
        return predicted_cls     
