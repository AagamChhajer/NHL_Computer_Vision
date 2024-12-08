import torch
import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class TPS_ResNet_BiLSTM_Attn(nn.Module):
    def __init__(self, opt):
        super(TPS_ResNet_BiLSTM_Attn, self).__init__()
        self.opt = opt

        # Transformation stage
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=opt.num_fiducial,
            I_size=(opt.imgH, opt.imgW),
            I_r_size=(opt.imgH, opt.imgW),
            I_channel_num=opt.input_channel
        )

        # Feature extraction stage
        self.FeatureExtraction = ResNet_FeatureExtractor(
            input_channel=opt.input_channel, 
            output_channel=opt.output_channel
        )
        self.FeatureExtraction_output = opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        # Sequence modeling stage
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size)
        )
        self.SequenceModeling_output = opt.hidden_size

        # Prediction stage
        self.Prediction = Attention(
            input_size=self.SequenceModeling_output, 
            hidden_size=opt.hidden_size, 
            num_classes=opt.num_class
        )

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(
            contextual_feature.contiguous(), 
            text, 
            is_train, 
            batch_max_length=self.opt.batch_max_length
        )

        return prediction
class Options:
    def __init__(self):
        self.Transformation = 'TPS'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'Attn'
        self.num_fiducial = 20
        self.imgH = 32
        self.imgW = 100
        self.input_channel = 3
        self.output_channel = 512
        self.hidden_size = 256
        self.num_class = 37  # Adjust based on the dataset (e.g., number of alphanumeric characters)
        self.batch_max_length = 25

opt = Options()
# Instantiate the model
model = TPS_ResNet_BiLSTM_Attn(opt)

# Load the state dictionary
state_dict = torch.load("./hello.pth")
model.load_state_dict(state_dict,strict=False)

# Switch to evaluation mode
model.eval()
