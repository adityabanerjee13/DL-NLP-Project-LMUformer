import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
from torch.utils.data import DataLoader

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    def __init__(self,
                 output_blocks=[3],
                 resize_input=True,
                 normalize_input=False,
                 requires_grad=False):
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        block1 = [
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        block2 = [
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        ]
        self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        block3 = [
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        ]
        self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
                break

        return outp
    
class inceptionScore:
    def __init__(self, data, device, batch = 500, cuda=True) -> None:
        self.model = InceptionV3()

        # for param in self.model.parameters():
        #     param.data = param.data.to(torch.float32)
        # device = next(self.model.parameters()).device  # Get device of model weights

        dataloader = DataLoader(data, batch_size=1000, shuffle=True, num_workers=4, pin_memory=True)
        for images, _ in dataloader:
            data = images.to(device)
            break
        self.batch = batch
        print('getting statistics of data...')
        self.m_data, self.s_data = self._compute_statistics_of_tensor(data, batch, cuda=cuda)
        print('complete !')
        
    def get_activations(self, samples, batch_size=50,
                        cuda=False, verbose=False,normalize=False, transpose=False, dims=2048):
        self.model.eval()

        if batch_size > samples.size(0):
            batch_size = samples.size(0)

        n_batches = samples.size(0) // batch_size
        n_used_imgs = n_batches * batch_size

        pred_arr = np.empty((int(n_used_imgs), int(dims)))
        with torch.no_grad():
            for i in tqdm(range(n_batches)):
                if verbose:
                    print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                        end='', flush=True)
                start = i * batch_size
                end = start + batch_size

                images = samples[start:end]
                
                if normalize and transpose:
                    # Reshape to (n_images, channel, height, width)
                    images = images.transpose((0, 3, 1, 2))
                    images /= 255

                batch = (images).type(torch.FloatTensor)
                if cuda:
                    batch = batch.cuda()

                pred = self.model(batch)[0]

                pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

        if verbose:
            print(' done')

        return pred_arr


    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)


    def calculate_activation_statistics(self, samples, batch_size=50,
                                        dims=2048, cuda=False, verbose=False):
        act = self.get_activations(samples, batch_size, cuda, verbose)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def _compute_statistics_of_tensor(self, samples, batch_size, cuda=False):
        m, s = self.calculate_activation_statistics(samples, batch_size, cuda)
        return m, s

    def calculate_fid_for_generatorSamples(self, samples, cuda=False):
        """Calculates the FID of two tensors"""
        print('calculating statistics of generated data...')
        m, s = self._compute_statistics_of_tensor(samples, batch_size=self.batch, cuda=cuda)
        fid_value = self.calculate_frechet_distance(m, s, self.m_data, self.s_data)
        return fid_value
