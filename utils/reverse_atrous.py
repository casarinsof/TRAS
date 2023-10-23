import torch


def reverse_atrous(input, stride):
    """
    stride is the equavalent of dilation in atrous convolution
    """
    #todo per semplicita' perche non dia errore se stride = 1 mi conviene restituire 4 copie del tensore..
    # capire se ha senso pero, sicuro e' inefficient
    batch_size, num_channels, d, d = input.size()
    R = stride
    subsampled_images = torch.zeros(batch_size, R * R, num_channels, d // R, d // R).cuda()
    for r in range(R * R):
        i, j = r // R, r % R
        subsampled_images[:, r, :, :, :] = input[:, :, i::R, j::R]

    if stride == 1:
        subsampled_images = torch.stack([input] * 4, dim=1)

    return subsampled_images

class SegmentConsensus(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output.squeeze(1)

class ConsensusModule(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)



if __name__ =='__main__':
    pass

    # Input tensor with shape (batch, channel, height, width)
    input_tensor = torch.zeros(1, 3, 12, 12)  # Example input tensor


    # Fill the tensor with a checkerboard pattern
    input_tensor[:, :, ::3, ::3] = 1  # Set even rows and even columns to 1

    input_tensor[:, 0, 1::3, ::3] = 0.2
    input_tensor[:, 1, 1::3, ::3] = 0.8
    input_tensor[:, 2, 1::3, ::3] = 0.9

    input_tensor[:, 0, ::3, 1::3] = 0.9
    input_tensor[:, 1, ::3, 1::3] = 0.2
    input_tensor[:, 2, ::3, 1::3] = 0.1

    input_tensor[:, 0, ::3, 2::3] = 0.4
    input_tensor[:, 1, ::3, 2::3] = 0.8
    input_tensor[:, 2, ::3, 2::3] = 0.2

    input_tensor[:, 0, 1::3, 1::3] = 0.8
    input_tensor[:, 1, 1::3, 1::3] = 0.8
    input_tensor[:, 2, 1::3, 1::3] = 0.2

    input_tensor[:, 0, 1::3, 2::3] = 0
    input_tensor[:, 1, 1::3, 2::3] = 0
    input_tensor[:, 2, 1::3, 2::3] = 1

    input_tensor[:, 0, 2::3, ::3] = 1
    input_tensor[:, 1, 2::3, ::3] = 0
    input_tensor[:, 2, 2::3, ::3] = 1

    input_tensor[:, 0, 2::3, 1::3] = 0.4
    input_tensor[:, 1, 2::3, 1::3] = 0
    input_tensor[:, 2, 2::3, 1::3] = 1

    import matplotlib.pyplot as plt

    plt.imshow(input_tensor[0].permute(1,2,0))

    plt.show()

    patches = reverse_atrous(input_tensor, stride=3)